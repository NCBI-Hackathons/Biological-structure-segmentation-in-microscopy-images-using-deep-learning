<!DOCTYPE html>
<html>

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ReadMe_September102018</title>
  <link rel="stylesheet" href="https://stackedit.io/style.css" />
</head>

<body class="stackedit">
  <div class="stackedit__html"><h1 id="nuclei-microscopy-image-segmentation">Nuclei Microscopy Image Segmentation</h1>
<h2 id="icmje-standard-citation">ICMJE Standard Citation:</h2>
<h3 id="doi-link">DOI link:</h3>
<p><img src="https://lh3.googleusercontent.com/DwtFufMw5tSa0h5Ia2S9JG0d6N3Ze3xEj4WcOnc7jPzdYSUrHX0pzmO1gnxUOpUAZEkCPq2X5m9QGQ" alt="examplestichingpostconnectivitymatrix" title="examplestitching"><br>
<strong>Figure 1.</strong> The stitching of the four regions of interest defined by the MASK-RCNN function.</p>
<h2 id="website-if-applicable">Website (if applicable)</h2>
<h2 id="intro-statement">Intro statement</h2>
<p>Finetuned convolutional neural networks (CNNs) have been traditionally used in the defense and security industries to classify objects of interest. However, increasingly, shallow CNNs such as AlexNet have already been used in pathology and radiology, namely to determine the malignancy of cancer tumors. Thus, biomedical scientists can readily exploit current networks designed for large-scale object recognition for microscopic biological systems.</p>
<h2 id="current-challenges">Current Challenges</h2>
<p>Since the advent of image processing, a gap remains in the physical techniques used to capture cell morphology and automated segmentation techniques used to quantify the data. While electron microscopy is a commonly used technique to image cell development, often manual methods are used to count the number and positions of cell types.</p>
<p>However, image workflows can vary greatly between laboratories depending on the microscope model and propietary software used to capture these images. This can cause variations in the pixel density and pixel grayscale intensity. Thus, traditional thresholding methods (e.g. Otsu’s Method, Histogram Thresholding, Gaussian Filtering) can be over-deterministic and only work on a case-by-case basis. Overall, current methods are cumbersome to the end-user and require weeks of preparation before implementation.</p>
<h2 id="why-should-we-solve-it">Why should we solve it?</h2>
<p>This would aim to provide a more robust, novel image processing pipeline that would decrease turnaround time for analyzing biological images en masse. Researchers could import grayscale electron microscopy Z-stack images in the form of .TIFF or .PNG files and view segmented nuclei images in a separte export folder. This would save both time and provide a more scalable means of nuclei segmentation.</p>
<h1 id="what-is-biological-structure-segmentation-in-microscopy-images-using-deep-learning">What is Biological-structure-segmentation-in-microscopy-images-using-deep-learning?</h1>
<p><img src="https://lh3.googleusercontent.com/YzIWqKFlvrYmNNssijs3dQbBAXqHvnAvs7fqTzlyfECOMoEXa7LenZt5N0BbH-U0iR81wDl_2CYXAQ" alt="workflowschematicofactualsoftware" title="workflowschematic"><br>
<strong>Figure 1.</strong> Overview digram of the workflow needed for image segmentation of electron microscopy images.</p>
<h1 id="how-to-use-biological-structure-segmentation-in-microscopy-images-using-deep-learning">How to use Biological-structure-segmentation-in-microscopy-images-using-deep-learning</h1>
<p>This software is based in Python 3 (Python 3.6) in Jupyter Notebook and relies on the compressed sparse graph routiness, scikit-learn and numpy packages. Ideally, a user could remote into a shared server such as BioWulf and run the script .py file after importing a folder of multiple high-resolution images.</p>
<h1 id="software-workflow-diagram">Software Workflow Diagram</h1>
<p><img src="https://lh3.googleusercontent.com/FybVO5MKiqwcoJQstrgiWAGI57nFibW-9nUDi_nR-Zz5EfyImdYhu-_GW4yCvwYXAL-hpvsxA1sgKA" alt="Software Workflow Diagram for Nuclei Segment" title="exampleworkflowforimprovednucleisegmentation"><br>
<strong>Figure 2.</strong> Schematic of the steps used in the improved nuclei segmentation software.</p>
<h1 id="file-structure-diagram">File structure diagram</h1>
<h4 id="define-paths-variable-names-etc"><em>Define paths, variable names, etc</em></h4>
<h1 id="installation-options">Installation options:</h1>
<p>We provide two options for installing Biological-structure-segmentation-in-microscopy-images-using-deep-learning: Docker or directly from Github.</p>
<h3 id="docker">Docker</h3>
<p>The Docker image contains Biological-structure-segmentation-in-microscopy-images-using-deep-learning as well as a webserver and FTP server in case you want to deploy the FTP server. It does also contain a web server for testing the Biological-structure-segmentation-in-microscopy-images-using-deep-learning main website (but should only be used for debug purposes).</p>
<ol>
<li><code>docker pull ncbihackathonsBiological-structure-segmentation-in-microscopy-images-using-deep-learning</code> command to pull the image from the DockerHub</li>
<li><code>docker run ncbihackathons/Biological-structure-segmentation-in-microscopy-images-using-deep-learning</code> Run the docker image from the master shell script</li>
<li>Edit the configuration files as below</li>
</ol>
<h3 id="installing-biological-structure-segmentation-in-microscopy-images-using-deep-learning-from-github">Installing Biological-structure-segmentation-in-microscopy-images-using-deep-learning from Github</h3>
<ol>
<li><code>git clone https://github.com/NCBI-Hackathons/Biological-structure-segmentation-in-microscopy-images-using-deep-learning.git</code></li>
<li>Edit the configuration files as below</li>
<li><code>sh server/Biological-structure-segmentation-in-microscopy-images-using-deep-learning.sh</code> to test</li>
<li>Add cron job as required (to execute Biological-structure-segmentation-in-microscopy-images-using-deep-learning.sh script)</li>
</ol>
<h3 id="configuration">Configuration</h3>
<p><code>Examples here</code></p>
<h1 id="testing">Testing</h1>
<p>We tested four different tools with Biological-structure-segmentation-in-microscopy-images-using-deep-learning They can be found in <a href="server/tools/">server/tools/</a> .</p>
<h1 id="additional-functionality">Additional Functionality</h1>
<h3 id="dockerfile">DockerFile</h3>
<p>Biological-structure-segmentation-in-microscopy-images-using-deep-learning comes with a Dockerfile which can be used to build the Docker image.</p>
<ol>
<li><code>git clone https://github.com/NCBI-Hackathons/Biological-structure-segmentation-in-microscopy-images-using-deep-learning.git</code></li>
<li><code>cd server</code></li>
<li><code>docker build --rm -t Biological-structure-segmentation-in-microscopy-images-using-deep-learning/Biological-structure-segmentation-in-microscopy-images-using-deep-learning .</code></li>
<li><code>docker run -t -i Biological-structure-segmentation-in-microscopy-images-using-deep-learning/Biological-structure-segmentation-in-microscopy-images-using-deep-learning</code></li>
</ol>
<h3 id="website">Website</h3>
<p>There is also a Docker image for hosting the main website. This should only be used for debug purposes.</p>
<ol>
<li><code>git clone https://github.com/NCBI-Hackathons/Biological-structure-segmentation-in-microscopy-images-using-deep-learning.git</code></li>
<li><code>cd Website</code></li>
<li><code>docker build --rm -t Biological-structure-segmentation-in-microscopy-images-using-deep-learning/website .</code></li>
<li><code>docker run -t -i Biological-structure-segmentation-in-microscopy-images-using-deep-learning/website</code></li>
</ol>
</div>
</body>

</html>
