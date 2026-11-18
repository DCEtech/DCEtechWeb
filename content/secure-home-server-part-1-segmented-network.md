Title: Building a Secure Home Server (Part 1): A Raspberry Pi with OpenWrt as the Gatekeeper
Date: 2026-11-18
Category: Cybersecurity
Tags: homelab, openwrt, raspberry-pi, network-segmentation, firewall, self-hosting
Slug: secure-home-server-part-1-segmented-network
Author: Daniel
Description: How I isolated my home server from my family's network with a Raspberry Pi 3B running OpenWrt as a firewall, mistakes included.
Summary: How I isolated my self-hosted server from my home network using a Raspberry Pi 3B running OpenWrt as a firewall, including every mistake I made along the way.

I have a small server at home. It runs a few Odoo instances in Docker, some demos for clients and a couple of personal tools. Sooner or later I want to publish some of those websites on the Internet, and that raises an uncomfortable question:

> If someone breaks into my server through one of those websites... what else can they reach?

On a typical home network the answer is: **everything**. Your laptop, your phone, your smart TV, the router's admin page. The server sits on the same network as all of them, so a single compromised web app becomes a door into your whole house.

This is the first post of a series where I build a home server that is **actually** secure, step by step, and at the end I attack it myself to prove it. In this first part I focus on the network: how to put the server in its own isolated zone, with a Raspberry Pi acting as the gatekeeper.

I'm a developer, not a network engineer, so I'll explain every concept the way I needed it explained. I'll also show the mistakes I made, because honestly they taught me more than the things that worked the first time.

## The series

1. **Network segmentation with a Raspberry Pi and OpenWrt** (this post)
2. Hardening the server: SSH keys, UFW, Docker and automatic security patches
3. Publishing websites without Cloudflare
4. Pentesting: attacking my own setup and publishing the results

## Why segment the network?

The idea is simple. Instead of one big network where everyone can talk to everyone, we create **two networks** and put a firewall between them:

- **Home network:** laptops, phones, TV. I consider it *untrusted* from the server's point of view.
- **Pro network:** only the server. This is the part exposed to the Internet, so it's the one most likely to get compromised one day.

The firewall enforces three rules:

1. The server **can** reach the Internet (updates, APIs...).
2. The server **cannot** reach anything in my home network.
3. From home, the only thing that can enter the pro network is web traffic (ports 80 and 443) and SSH from **one** specific computer: mine.

The best analogy I found is a building with a doorman and two doors: a main door to the street, which is watched and mostly locked, and an inner door to the office. The doorman (the Raspberry Pi) decides who goes where.

## The architecture

```text
[INTERNET]
    │
    ▼
[ISP router] ── home network 192.168.1.0/24 (laptops, phones, TV)
    │
    │ cable → Pi built-in Ethernet port (WAN)
    ▼
[Raspberry Pi 3B + OpenWrt] ── the "doorman"
    │  eth0 (built-in)   = WAN  → 192.168.1.50 (home side)
    │  eth1 (USB adapter) = LAN → 10.10.10.1/24 (pro side)
    │  Port forwarding: 80, 443 → 10.10.10.10
    │
    │ cable → USB adapter
    ▼
[Ubuntu server] ── pro network 10.10.10.0/24
       10.10.10.10, Docker containers listening only on 127.0.0.1
```

A Raspberry Pi 3B only has one Ethernet port, so I added a USB Ethernet adapter to get the second "door". One port faces the ISP router, the other faces the server.

**Hardware limitations to be honest about:** the Pi 3B's built-in port is 100 Mbps and shares the USB 2.0 bus with the adapter. For a few websites at home it's more than enough, but it's not a gigabit firewall. If you're buying hardware specifically for this, look at a dedicated router or a newer board.

### A note for people in Spain: why no Cloudflare

Many homelab guides tell you to put Cloudflare in front of everything. In Spain, during football matches, ISPs have been blocking whole IP ranges (including Cloudflare's) as part of LaLiga's anti-piracy actions, which takes down completely unrelated websites. So my setup uses **direct DNS** (no proxy) with dynamic DNS, and a public IPv4 address from my ISP instead of sitting behind CG-NAT. I'll cover this in part 3.

## Step 1: Building the OpenWrt image

[OpenWrt](https://openwrt.org/) is a Linux distribution for routers. Instead of downloading the generic image, I used the official **Firmware Selector**, which lets you build a custom image with the packages you choose.

My choices:

| Decision | Choice | Why |
|---|---|---|
| Version | OpenWrt 25.12 | Current stable branch. Note: 25.12 replaced `opkg` with **`apk`** as the package manager. |
| Architecture | 64-bit (`bcm27xx/bcm2710`) | The Pi 3B's CPU is ARMv8; 64-bit is its native mode. |
| Image type | `squashfs-factory` | Read-only root filesystem with a `firstboot` factory reset. More robust against power cuts than `ext4`. |
| Added packages | `kmod-usb-net-rtl8152`, `r8152-firmware`, `usbutils` | The driver for the USB adapter, baked into the image (more on this below). |
| Removed packages | Wi-Fi drivers and `wpad`, PPPoE, audio | **A firewall should not have a radio.** If someone accidentally enables a Wi-Fi access point, it bypasses the whole segmentation. What isn't installed can't be misconfigured. |

**The chicken-and-egg problem:** without the USB adapter's driver there is no second port, and without a configured WAN there is no Internet to download the driver. That's why I added the driver to the image itself.

Flashing the SD card from Linux:

```bash
# Verify the download first (compare with the hash shown by the Firmware Selector)
sha256sum openwrt-*-rpi-3-squashfs-factory.img.gz

# Identify the SD card: run lsblk before AND after inserting it
lsblk -o NAME,SIZE,TYPE,MODEL,MOUNTPOINTS

# Write it (replace sdX with YOUR card — dd will wipe whatever you point it at)
gunzip -c openwrt-*-rpi-3-squashfs-factory.img.gz | \
  sudo dd of=/dev/sdX bs=4M conv=fsync status=progress
```

> **Mistake #1:** my first download was the *generic* image, not my custom build. The Firmware Selector shows the generic files as soon as you choose the device; your custom image only exists after you click "Request build" and wait. I only noticed because I inspected the image before flashing it. Check that the filename includes a build hash.

## Step 2: First boot (isolated!)

Out of the box, OpenWrt behaves like a brand new home router: it takes the address **192.168.1.1** and starts a **DHCP server**. That's the same address as most ISP routers. If you plug it into your home network you get an IP conflict, and your family's devices might start getting addresses from the Pi instead of the real router and lose Internet.

So the first boot happens with a cable **directly from my PC to the Pi**, and the PC's Wi-Fi turned off (otherwise the PC would be on two different networks that both use 192.168.1.x).

```bash
ssh root@192.168.1.1
passwd                      # first thing: set a root password
```

Then I moved the Pi's internal network to its final range, so it never collides with the home network again.

> **Mistake #2 (the big one):** in OpenWrt 25.12, IP addresses are stored as a **list in CIDR notation** (`192.168.1.1/24`), not as address + netmask. I set the new address without the `/24`:
>
> ```sh
> uci set network.lan.ipaddr='10.10.10.1'   # WRONG on 25.12
> ```
>
> The Pi ended up with a `/32`, a network of exactly one address. The symptoms were confusing: my PC could see the Pi at the ARP level (`ip neigh` showed its MAC), but ping, SSH and DHCP all failed, because the Pi had no route back to anyone.
>
> The way back in was **IPv6 link-local**: every interface gets an automatic `fe80::` address that works on a direct cable regardless of IPv4 configuration.
>
> ```bash
> # Find devices on the cable
> ping -6 -c 3 ff02::1%enp39s0
> # SSH using the Pi's link-local address
> ssh root@fe80::xxxx:xxxx:xxxx:xxxx%enp39s0
> ```
>
> And the correct way to set the address:
>
> ```sh
> uci delete network.lan.ipaddr
> uci add_list network.lan.ipaddr='10.10.10.1/24'
> uci commit network
> ```
>
> **Lesson: always run `uci show` and look at how the configuration is actually stored before changing it.**

After that: hostname, timezone, an **ed25519 SSH key** instead of a password, and a configuration backup with `sysupgrade -b`. Backups of a known-good state are what let you experiment without fear.

## Step 3: The USB adapter that wasn't what it claimed to be

I bought a "Realtek RTL8153" gigabit USB adapter, which is the usual recommendation because Linux supports it well. When I plugged it into the Pi, it kept disconnecting and reconnecting every five seconds, and no `eth1` ever appeared.

`lsusb` gave it away:

```text
ID 38a2:0153 TIH Corp USB NetCard
```

A genuine RTL8153 identifies itself as `0bda:8153` (`0bda` is Realtek's vendor ID). This was a **clone** with an unknown vendor ID, so the Realtek driver ignored it. Linux picks drivers by that ID, and when nobody claimed the device it kept resetting itself.

To find out what it really was, I plugged it into my Ubuntu PC, which ships with far more drivers:

```bash
lsusb -t     # shows which driver claimed each USB device
```

It turned out to be a device that speaks **RNDIS** (a Microsoft protocol, the same one Android uses for USB tethering) and **CDC Ethernet**, the standard USB networking class. On OpenWrt I needed `kmod-usb-net`, `kmod-usb-net-cdc-ether` and `kmod-usb-net-rndis`.

The Pi still had no Internet, so I downloaded the packages on my PC from the exact kernel modules folder of my release (`cat /etc/apk/repositories.d/distfeeds.list` on the Pi tells you which one), copied them over with `scp -O`, and installed them offline:

```sh
apk add --no-network --allow-untrusted kmod-usb-net*.apk
```

Two things worth knowing:

- **Kernel modules must match the exact kernel build.** A module from a slightly different version will not load. Always use the folder your device points to.
- `apk` verifies packages through the **signed repository index**, not individually. Offline, it can't check that index, so it reports `UNTRUSTED signature`. `--allow-untrusted` is acceptable here because I downloaded the files myself over HTTPS from the official server. For a client I'd go the strict way: download the signed index too and use it as a local repository.

With the drivers in place the adapter came up as `eth1`, using the standard `cdc_ether` driver, and it stayed stable after reboots.

> **Lesson:** cheap "branded" network adapters are often clones. Check the USB ID (`lsusb`) as soon as it arrives. For production, buy hardware with a chip you can identify.

## Step 4: Two zones, two doors

OpenWrt's firewall works with **zones**. By default there are two:

- `lan`: trusted, everything allowed.
- `wan`: the outside, incoming traffic rejected.

In my design, `wan` is my **home network** and `lan` is the **pro network**. The steps, in an order that never left me without a way in:

1. Add `eth1` (the adapter) to the LAN bridge **without removing** `eth0`, so both ports serve the pro network for a while.
2. Move my PC's cable to the adapter and confirm I can still log in.
3. Only then remove `eth0` from the bridge and turn it into the WAN.

```sh
uci add_list network.@device[0].ports='eth1'
# ...test access through the adapter...
uci del_list network.@device[0].ports='eth0'
uci set network.wan=interface
uci set network.wan.device='eth0'
uci set network.wan.proto='static'
uci add_list network.wan.ipaddr='192.168.1.50/24'
uci set network.wan.gateway='192.168.1.1'
uci add_list network.wan.dns='9.9.9.9'
uci add_list network.wan.dns='149.112.112.112'
uci changes network     # review BEFORE committing
uci commit network
service network restart
```

Some decisions here that are easy to get wrong:

- **Static IP outside the router's DHCP pool.** My first choice was `.250`, which "looked high enough". When I checked the ISP router, its pool was `.128` to `.254`, so `.250` would eventually have been handed to my phone. Check the pool, don't assume. Addresses `.2` to `.127` were free.
- **Don't depend on the ISP router's configuration** when you can avoid it. ISP technicians reset those routers without asking. A static IP outside the default pool survives a factory reset; a shrunk DHCP pool doesn't.
- **DNS:** I used Quad9 instead of Cloudflare's resolvers, for the same reason as above.
- Check that OpenWrt **won't serve DHCP on the WAN** (`uci show dhcp.wan` should show `ignore='1'`).

> **Mistake #3:** when I moved the Pi next to the ISP router, I plugged the router into the **USB adapter** instead of the built-in port. For a few minutes the pro network, with its DHCP server and IPv6 announcements, was inside my home network. I later found IPv6 addresses from the Pi's internal prefix on my server: the evidence. **Built-in port → ISP router. Adapter → server.** Label your cables.

## Step 5: Managing the Pi from another room

My PC is in a different room from the router, so the direct cable was no longer an option. The firewall blocks SSH from the home network on purpose, so I opened exactly **one** exception: SSH from my PC's IP address.

That required my PC to always have the same address, so I created a **DHCP reservation** on the ISP router (MAC → IP), and then the firewall rule on the Pi:

```sh
uci add firewall rule
uci set firewall.@rule[-1].name='Allow-SSH-Admin-PC'
uci set firewall.@rule[-1].src='wan'
uci set firewall.@rule[-1].src_ip='192.168.1.132'
uci set firewall.@rule[-1].proto='tcp'
uci set firewall.@rule[-1].dest_port='22'
uci set firewall.@rule[-1].target='ACCEPT'
uci commit firewall
service firewall restart
```

Firewall changes only take effect after `uci commit firewall` and `service firewall restart`. I leave those two lines out of the next blocks to keep them short.

> **Mistake #4:** the next day, SSH said `Connection refused`. My PC had a different IP. The reservation on the router had a typo in the MAC address (two hex pairs swapped), so the router silently ignored it, and the day before I had simply been lucky enough to get the right IP. **Never type MAC or IP addresses by hand. Copy and paste.**

## Step 6: Turning the Pi into a bunker

This is the core of the whole setup.

### 6.1 The server can't reach home

OpenWrt's defaults let `lan` forward to `wan`, because that's how a LAN reaches the Internet. But in my design `wan` is also my home network, so I added an explicit block:

```sh
uci add firewall rule
uci set firewall.@rule[-1].name='Block-LANPRO-to-Home'
uci set firewall.@rule[-1].src='lan'
uci set firewall.@rule[-1].dest='wan'
uci set firewall.@rule[-1].dest_ip='192.168.1.0/24'
uci set firewall.@rule[-1].proto='all'
uci set firewall.@rule[-1].target='REJECT'
```

A common doubt: *if the server reaches the Internet through the home router, doesn't this cut its Internet access?* No. The firewall looks at the **final destination** of each packet. A request to a website has that website as its destination; the router is just the path.

### 6.2 The server can't reach the Pi itself

This one is easy to miss. The `lan` zone is marked as trusted (`input ACCEPT`), which makes sense in a normal home router. But here, the LAN is the zone exposed to the Internet. If an attacker took over the server and then reached the Pi's SSH or web interface, they could **turn off the firewall**. The Pi is the master key, so it must be unreachable.

So: reject everything from the pro network to the Pi, except the three things the server genuinely needs.

```sh
# Allow only what the server needs from the Pi
uci add firewall rule
uci set firewall.@rule[-1].name='Allow-LAN-DHCP'
uci set firewall.@rule[-1].src='lan'
uci set firewall.@rule[-1].proto='udp'
uci set firewall.@rule[-1].dest_port='67 547'
uci set firewall.@rule[-1].target='ACCEPT'

uci add firewall rule
uci set firewall.@rule[-1].name='Allow-LAN-DNS'
uci set firewall.@rule[-1].src='lan'
uci set firewall.@rule[-1].proto='tcp udp'
uci set firewall.@rule[-1].dest_port='53'
uci set firewall.@rule[-1].target='ACCEPT'

uci add firewall rule
uci set firewall.@rule[-1].name='Allow-LAN-ICMP'
uci set firewall.@rule[-1].src='lan'
uci set firewall.@rule[-1].proto='icmp'
uci set firewall.@rule[-1].target='ACCEPT'

# Then close the door
uci set firewall.@zone[0].input='REJECT'
```

The order matters: first the exceptions, then close the door, so the server never loses DNS in between.

> **Mistake #5:** in my notes, the "how to undo this" commands were right below the "apply" commands. I copied one line too many and committed `input='ACCEPT'`, so the door stayed open. Keep rollback instructions clearly separated from the apply steps.

### 6.3 Everything else

- **SSH with keys only** on the Pi (Dropbear: `PasswordAuth` and `RootPasswordAuth` off), tested from a *second* terminal while keeping the first session open.
- **Disabled default rules I don't use**, such as the IPSec VPN pass-through rules. Disabled, not deleted, so they're easy to restore.
- **Port forwarding (DNAT)** for 80 and 443 to the server. OpenWrt's fw4 also creates NAT reflection rules automatically.
- The **LuCI web interface** is only accessed through an SSH tunnel, never exposed.

## Step 7: Moving the server behind the Pi

The cleanest trick of the whole project: the server already got its IP via DHCP. Instead of editing its network configuration, I created a **DHCP reservation on the Pi** for the server's MAC address, and then simply moved its cable from the ISP router to the Pi's adapter. The server asked for an address, the Pi gave it `10.10.10.10`, done.

The rollback plan was just as simple: plug the cable back into the ISP router.

To reach the server from my PC, I didn't open any new port. SSH's `ProxyJump` uses the Pi as a stepping stone:

```text
# ~/.ssh/config on my PC
Host server
    HostName 10.10.10.10
    User myuser
    ProxyJump pi
    IdentityFile ~/.ssh/id_ed25519_server
```

`ssh server` connects to the Pi (already allowed), and from the Pi to the server.

## Verification: don't assume, prove

The tests I ran from the server, with the actual results:

| Test from the server | Expected | Result |
|---|---|---|
| `ping 9.9.9.9` | Internet works | ✅ replies |
| `ping` to my PC | Blocked | ❌ `Destination Port Unreachable` **from 10.10.10.1**, so the Pi is the one stopping it |
| `ping` to the ISP router | Blocked | ❌ same |
| `nslookup` a domain | DNS works | ✅ resolves |
| `nc -zv 10.10.10.1 22` (Pi's SSH) | Blocked | ❌ `Connection refused` |
| `nc -zv 10.10.10.1 80` (Pi's web UI) | Blocked | ❌ `Connection refused` |

On the Pi, the block rule has a packet counter, and it went up by exactly the number of packets I tried to send:

```sh
nft list ruleset | grep 'Block-LANPRO'
# ... counter packets 6 bytes 504 jump reject_to_wan ...
```

In part 4 I'll do this formally from three positions (from the Internet, from a "compromised" device at home, and from a "compromised" server) and publish the results.

## Lessons learned

1. **Look at the real configuration before changing it.** Versions change formats (`/24` in OpenWrt 25.12).
2. **Always keep a way back in.** A second terminal, a direct cable, IPv6 link-local, a backup from `sysupgrade -b`, and a keyboard and screen as a last resort.
3. **Copy and paste addresses.** A MAC typo fails silently.
4. **Check, don't assume:** DHCP pools, free IPs (`ping` before using one), which physical port is which.
5. **Least privilege, also towards your own firewall.** The network behind the firewall is the one exposed to the Internet; don't trust it.
6. **What isn't installed can't be attacked.** No Wi-Fi on the firewall, no unused VPN rules.
7. **Verify with tests**, including the ones that are supposed to fail.

## What's next

In **part 2** I harden the server itself, because the Pi is the wall but the house still needs a lock: SSH with ed25519 keys only, UFW (and why Docker bypasses it), keeping containers bound to `127.0.0.1`, removing unnecessary services, and automatic security patches.

If you're building something similar and get stuck, feel free to get in touch through the contact page.
