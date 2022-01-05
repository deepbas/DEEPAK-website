---
title: Using Rstudio in Stat 220
draft: false  
toc: false  
type: book  
weight: 20
---

Key points

There will be a lot of Rstudio content thrown your way this term, most
in the form of .Rmd Markdown files. To stay organized, I *strongly*
suggest you create a **stat220** folder that contains the following
subfolders:

-   **stat220** folder
    -   **Assignments:** This folder will contain subfolders for each
        assignment. Each assignment subfolder (e.g. homework1,
        homework2, …) will be a Github connected RStudio project that
        you will create **once an assignment is posted.**

    -   **Content:** This folder should be used to save any
        non-assignment files (e.g. slides, examples) for this class. You
        will create this subfolder by creating an RStudio project (see
        step 5 below).

To get started with this organization, follow the steps below.

------------------------------------------------------------------------

# File organization: Using maize

The server (online) version of Rstudio is run from a unix server. You
can navigate this file system using unix commands, but I assume that
most or all of you will just use Rstudio to access your files on this
server.

-   **1.** In Rstudio, click the **Files** *tab* in the lower right-hand
    window. Note: this is **not** the same as the **File** *menu*
    option.

![](/img/maize_files.png)

-   **2.** Verify that you are in your **HOME** folder (should simply
    say Home right under the New Folder button). To navigate to your
    Home folder (if somehow you are not in it), click the **…** button
    (far right side of the **Files** tab) and enter a \~ (tilde) symbol

![](/img/maize_home.png)

-   **3.** Click the **New Folder** button and name the folder
    **stat220**.

![](/img/maize_newfolder.png)

-   **4.** Click on this newly created (empty) **stat220** folder.
    Within the folder create another **New Folder** and name it
    **assignments**.

![](/img/maize_stat220.png)

-   **5.** Within the **stat220** folder, create an **RStudio project**
    called **content** with the following steps:

    -   **a.** Click the **Project** button in the upper righthand
        corner of your RStudio window and select **New Project…**.

![](/img/maize_project.png)

-   **b.** Select **New Directory** and then **New Project**

![](/img/maize_newdirectory.png)

![](/img/maize_newdirectory2.png)

-   **c.** Enter **content** as the **Directory name** and use the
    **Browse** button to find your **stat220** folder. Then click
    **Create Project**.

![](/img/maize_create.png)

-   **d.** You should now have a new folder called **content** in your
    **stat220** folder and this folder will contain an RStudio project
    `.Rproj`. Feel free to add subfolders to this **content** folder
    (e.g. slides, examples, etc).

![](/img/maize_Rproj.png)

**Warning: Do not** create an RStudio project in the main stat220 folder
because it is not good practice to have RStudio projects in subfolders
of another project (e.g. a project within a project is not recommended).

------------------------------------------------------------------------

# File organization: Using your own Rstudio

Create a folder called **stat220** somewhere on your computer. Within
this folder create an **assignments** subfolder. Then complete **step
5** from above to create a **content** RStudio project folder.

------------------------------------------------------------------------

# RStudio projects

Once you’ve created a project, your R session should be running within
that project folder. You can check which project you are in by checking
the project name in the upper righthand part of your RStudio window.
Here we see the **content** project is open:

![](/img/maize_content.png)

Running R from an RStudio project sets your **working directory** to the
project folder:

![](/img/maize_getwd.png)

This allows for easy file path access to all files related to this
project.

To **start** a project, click on the `.Rproj` file or use the **Open
Project…** option shown in step 5 above.

------------------------------------------------------------------------

# Best practices (or what not to do)

-   Never save files to a lab computer hard drive (e.g. desktop,
    downloads, etc). They will be erased when you log off.
-   Do not use gmail as a file storage system! Avoid emailing yourself
    files that you created (and saved) on a lab computer. Eventually you
    will lose work this way.
-   Avoid using online versions of google drive and dropbox. Similar to
    gmail, downloading, editing a doc, then uploading it back to
    drive/dropbox is another great way to lose work.
-   Avoid [this](https://xkcd.com/1459/) and [this](http://phdcomics.com/comics.php?f=1531).
