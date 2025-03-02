Title: How to use Pelican: Markdown files, theme selection and customization.
Date: 2024-09-12
Category: WebDevelopment
Tags: pelican, python, blog, web
Author: Daniel

<br>

In the previous post you have the introduction about how to start with pelican, and in this one I´m going to share 
more info about how to use pelican.

<br>

<span class=sub-title>
 Markdown Files to generate articles: 
</span>

<br>

One of the main reasons I choose Pelican, I can use a simple files to generate my content with a light and simple syntax.
How can I make an article in pelican? Simple, create a markdown file that name ends in .md or mk in the content directory and
use this structure in your md file: 

<br>
  <span class="boxed-text">
   <span class="red-text">
      Title:     <br>
      Date:      <br>
      Category:  <br>
      Tags:      <br>
      Author:    <br>
      Content goes here..
   </span>
  </span>

<br>

<span class=sub-title>
 Theme Selection:
</span>

<br>

One of the most powerful features of Pelican it´s that you can choose different themes, this themes are opensource that the
pelican community provides, you can find different themes in this link [Pelican themes](https://pelicanthemes.com/).
If you want to change the theme that pelican have by default, here you have some steps: 

<br>

1º Create a directory named themes in your pelican project and clone the repository that the theme you choose:

<br>
  <span class="boxed-text">
   <span class="green-text">
     mkdir themes <br>
     cd themes    <br>
     git clone https://github.com/username/repository-name.git <br>
   </span>
  </span>

<br>

2º Configure pelican by the new theme in file pelicanconf.py:

<br>
  <span class="boxed-text">
   <span class="red-text">
     THEME = 'themes/repository-name'
   </span>
  </span>

<br>

3º Generate you site with the new theme: 

<br>
  <span class="boxed-text">
   <span class="green-text">
      pelican content  <br>
      pelican --listen
   </span>
  </span>

<br>

<span class=sub-title>
 Customize our own theme and what is Jinja2?
</span>

<br>

![img.png]({static}/images/jinja2.png)

<br>

Up to this point it hasn´t been necessary nothing to code in HTML or CSS. However, if you want to customize your own 
theme, you´ll need to have some basic knowledge of HTML, CSS and jinja2. I assume you´re familiar with HTML and CSS, 
but you might not know much about Jinja2. Let me explain: Jinja2 is an extensible template engine used in Python that 
allows developers to create HTML templates that can be reused and filled with dynamic content. In Pelican, Jinja2 is 
specifically used to inject content into templates and render them into HTML files during Pelican´s build process. 
What does this mean? It means that Pelican uses this technology to manage the HTML and the markdown files to produce the 
final HTML that generates the website. 
If you are interested to learn more about Jinja2 visit their website [Jinja](https://jinja.palletsprojects.com/en/3.1.x/).

<br>

Ok, with this brief introduction about Jinja2 let´s continue to customize our own theme. To create and customize
you own theme it´s easy if you have the basic structure to work in it, in the Pelican website suggest to use the simple
theme to create our own theme, this is our basic structure. Here are some steps:

<br>

1º Install pelican themes:

<br>
  <span class="boxed-text">
   <span class="gree-text">
     pip install pelican themes
   </span>
  </span>

<br>

2º Go to Pelican project directory venv/lib/site-packages/pelican/themes/simple and copy the simple theme in our
themes project directory that we created before. I recommend to rename the directory for something like mytheme or 
similar to this.

<br>

3º Configure the pelicanconf.py with the new renamed theme: 

<br>
  <span class="boxed-text">
   <span class="red-text">
     THEME = 'themes/mytheme'
   </span>
  </span>

<br>

4º Generate the site, to view the changes:

<br>
  <span class="boxed-text">
   <span class="green-text">
      pelican content  <br>
      pelican --listen
   </span>
  </span>

<br>

If you followed the steps correctly you might view also like this:

<br>

![img.png]({static}/images/simple.png)

<br>

Ok, this is really simple now it´s your turn to play with HTML and CSS to customize and create a nice website, just a 
few pieces of advices be careful with the project directory, as it´s very sensitive for any change, use the Jinja2 it´s really 
useful and have a happy coding. 
