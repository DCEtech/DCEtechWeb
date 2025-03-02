Title: Hello World - How I Built This Website?
Date: 2024-09-12
Category: WebDevelopment
Tags: pelican, python, blog, web
Author: Daniel


<br>

  The answer is simple: I used Pelican. But what exactly is Pelican?


<br>

![img.png]({static}/images/pelican.png)

<br>

Pelican is a static site generator that allows you to create websites quickly and easily without needing to write HTML 
or CSS. Instead, you can choose from a variety of themes provided by the Pelican community. Pelican is written in 
Python, which I consider one of the best programming languages due to its simplicity and speed. With Pelican, you can 
write your content in Markdown and use a simple CLI tool to generate your site. These are just a few advantages of 
Pelican; you can find more information on the [Pelican website](https://docs.getpelican.com/en/latest/#).

<br>
It’s clear why I chose Pelican to build my site. But I haven’t yet explained how I actually made it!

<br>
The process was straightforward. After choosing Pelican, the next step was selecting a theme. However, I didn’t find 
any that I particularly liked, so I decided to customize my own theme. I based it on the “simple” theme, which is 
specifically designed to be a starting point for creating custom themes. The simple theme provides a structure that you
can modify to create a design that suits your needs. 

<br>
Great! Now, let me walk you through the first steps to generate a Pelican site:

<br>

1º Set up a Python virtual environment for your project:

<br>
  <span class="boxed-text">
   <span class="green-text">
      mkdir pelican-site && cd pelican-site  <br>
      python3 -m venv venv
   </span>
  </span>

<br>

2º Activate the virtual environment:  

<br>

  <span class="boxed-text">
    <span class="green-text">
      cd venv/Scripts <br>
      activate
    </span>
  </span>

   

<br>

3º Install Pelican:  

<br>

  <span class="boxed-text">
    <span class="green-text">
      pip install "pelican[markdown]"
    </span>
  </span>

<br>

4º Run the quickstart setup. This will let you configure the generator to fit your needs:

<br>
  <span class="boxed-text">
    <span class="green-text">
      pelican-quickstart
    </span>
  </span>

<br>

   Now the site generator will ask you a series of questions to set up as your preferences:

<br>

 <span class="boxed-text">
    <span class="green-text">
      Where do you want to create your new web site? [.] <br>
      What will be the title of this web site? <br>
      Who will be the author of this web site? <br>
      What will be the default language of this web site? [es] <br>
      Do you want to specify a URL prefix? e.g., https://example.com   (Y/n) <br>
      What is your URL prefix? (see above example; no trailing slash) <br>
      Do you want to enable article pagination? (Y/n) <br>
      How many articles per page do you want? [10] <br>
      What is your time zone? [Europe/Rome] <br>
      Do you want to generate a tasks.py/Makefile to automate generation and publishing? (Y/n) <br>
      Do you want to upload your website using FTP? (y/N) <br>
      Do you want to upload your website using FTP? (y/N) <br>
      Do you want to upload your website using SSH? (y/N) <br>
      Do you want to upload your website using Dropbox? (y/N) <br>
      Do you want to upload your website using S3? (y/N) <br>
      Do you want to upload your website using Rackspace Cloud Files? (y/N) <br>
      Do you want to upload your website using GitHub Pages? (y/N) <br>
   </span>
 </span>

<br>

We’ve successfully generated our site with Pelican. If you would like to view the webiste, use these CLI comands:

<br>

<span class="boxed-text">
    <span class="green-text">
       pelican content <br>
       pelican --listen
    </span>
</span>

<br>

By default the pelican generator use the port 8000 to serve the website. Just copy and paste the url that Pelican provides 
into your browser, and you should see something like this: 

<br>

![img.png]({static}/images/example.png)

<br>

And that’s it! Remember this is just a simple example. In the next post, we’ll explore more actions we can take with 
Pelican, such as creating pages and articles with Markdowns, using various themes, and customizing our own.
   

