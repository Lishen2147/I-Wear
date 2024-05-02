Group member names: Steve Chen, Brian Li, Lishen Liu, Anh Nguyen, Megan Nguyen,
Ryan Yu
# ECS 170 Project Proposal
## Project Name: **I-Wear Recommender**
## High-Level Overview:
- This project would consist of a web application that takes in real-time taken
  or user uploaded images of the user and determines their facial shape. After,
  accessories like glasses are recommended to the user based on research done by
  prior beauticians. Though this web application is new, the idea of
  complementing your natural beauty is not. Currently, there are many trends on
  social media where people are trying to use science to enhance natural beauty.
  Whether that be through color analysis, proportional outfits, or makeup,
  people want to enrich their innate features with the use of AI/scientific
  tools. Our target audiences are glasses companies to utilize this tool for
  their customers to have an easy way to choose glasses that complement their
  features.
## How we will utilize AI:
- This project uses supervised learning and computer vision to detect and
  recognize aspects of a user’s face.
## Details, Implementation, Scaffolding, etc.:
- To determine the face shape of individuals in the images, we will implement a
  convolutional neural network using PyTorch or use a built-in CNN. This CNN
  model performs a classification task onto the given image, suppose this image
  represents a face of a human, and it will eventually classify this face into
  certain labels such as oval, square, round, pear, oblong, etc. This output
  will further help us to connect with the next step, which is the shape (type)
  of glasses that best suit our client’s facial features. Since this is a
  classification task, we will use accuracy, precision, recall, and F1 score as
  evaluation metrics. Instead of implementing these evaluation metrics, we will
  take use of the given “scikit-learn” package in Python.
- Through experimentation, we will settle on specific CNN parameters (kernel
  size, stride, etc.), and parameters in any neural network (learn rate, batch
  size, number of epochs, etc.). We will try to implement mini-batch gradient
  descent in order to optimize training and memory costs. For our computing
  resources, we will use our own computers and cloud services like Google CoLab
  and Amazon SageMaker.
- For our scaffolding, we will use a face dataset from Kaggle, which contains
  5000 images of female celebrities from all around the globe which are
  categorized according to their face shape. The face shape features include:
  Heart, Oblong, Oval, Round, and Square. In addition, we plan to use the Specs
  on Faces (SoF) dataset from Papers With Code, which is a collection of 42,592
  images for 112 people (66 males and 46 females) who wear glasses under
  different illumination conditions.
- We will use standard web technologies for our frontend and backend. We will
  leverage express.js, a server API package, to help users interact with our
  model on the cloud. For the frontend, we will use React.js to write the
  necessary JSX code to populate a web page with frontend components. We will
  also use Redux and Redux-saga to effectively store and query our backend API
  for model predictions.
- For the frontend, we will use:
    - React.js
    - Redux
    - Redux Saga
    - Other component libraries
- For the backend, we will use:
    - Express
    - Axios
    - Other backend libraries
