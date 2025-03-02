Title: My Final Degree Project
Date: 2025-03-01
Category: MultiplatformDevelopment
Tags: dataAnalysis, dataVisualization, webDevelopment, mobileDevelopment, blog
Author: Daniel

<br>

<span class=sub-title>
Introducing Financiero: A Multiplatform Finance App
</span>

<br>

Fist of all I´m going to introduce my multiplatform app, Financiero. You can check out this project on GitHub [Financiero](https://github.com/DCEtech/financiero).

<br>

It is a simple but powerfull app that allows you register your financial data and track you savings progress. One of my main motivations for developing this app is the importance of financial management. I belive that visualizing our savings growth can inspire people to save more and improve their financial habits.

<br>

<span class=sub-title>
Technologies Used
</span>

<br>

<span class=sub-title>
Multiplatform Development
</span>

<br>

To build both a web and mobile app, I needed a multiplatform framework. Since I really like Python, I chose Flet, a Python-based framework build on Flutter. It´s powerfull, simple, and easy to use. 

<br>

<span class=sub-title>
Database
<span>

<br>

For data storage I use a NoSQL data base: 
MongoDB. I also like working with JSON, which MongoDB stores as BSON. This database offers great fexibility and high performance, making it ideal for handling financial data.

<br>

<span class=sub-title>
Deployment with Docker
<span>

<br>

To enhance deployment, I used Docker to containerize MongoDB. Learning DevOps skills is essential when deploying projects. In this case I use Docker Compose, wich is perfect for development or testing enviroments. 

<br>

All you need is a docker-compose.ymal file, that describes all the cointainers are part of the app:

<br>

![img.png]({static}/images/dockerCompose.png)

<br>

<span class=sub-title>
Data Visualization
<span>

<br> 

For data visualization, I used the well-known Matplotlib library to create charts showing savings growth.

<br>

<span class=sub-title>
User Authentication and Security 
<span>

<br>

The app includes a login system, which is managed in the database. To ensure strong and secure passwords,  I encrypt the user passwords before storing them. When a user logs in, the system  decrypts the stored password and comprares it with the input to ensure privacy and securty.
<br>
I use bcrypt, a password hashing function designed to be secure against brute-force attacks. This hashing  method includes:
 
- Salt generation: Adds randomness to prevent rainbow table attacks. 
- Key stretching: Increases computational cost, making brutal-force attacks harder. 

<br>

<span class=sub-title>
User Registration with Password Hashing
<span>

<br>

Her´s the function to create a user with a hashed password: 

<br>

![img.png]({static}/images/createUser.png)

<br>

<span class=sub-title>
User Authentication 
<span>

<br>

This function authenticates a user when they log in: 

<br>

![img.png]({static}/images/authenticationUser.png)

<br>

<span class=sub-title>
Final Thoughts
<span>

<br>

With Financiero, I aimed to create a simple yet prowerful finance management app that helps users track thir savings efficiently. By combining Python, Flet, MongoDB, Docker and Matplotlib, I build a cross-platform application with secure authetiucation and real-time financial tracking.

<br>

I hope this project inspires others to manage thir finances better and explore new technologies in development. 

