# Style-Transfer
This is a piece of code which optimizes a given( if not given then random) image into something which takes the content from one image and style of another.
<br/>
I have used just the Tensorflow Keras backend so that the computations could be done with the help of multiple threads (or GPU) and used scipy's **fmin_l_bfgs_b** which just implements an advanced version of newtons formula for roots. But rather here it finds roots for the derivative of the loss function which should corresponds to its minima (most probably).
<br/>
## Demo
I have tried keeping the same content to experiment with the hyperparameters of the style loss.
<table>
<tr>
	<th>Sl no.</th>
	<th>Content</th>
	<th>Style</th>
	<th>Output</th>
</tr>
<tr>
	<td><a id="first">1</a></td>
	<td><img src="/content.jpg" width="200" /></td>
	<td><img src="/style.jpg" width="200" /></td>
	<td><img src="/outputs/combined2.jpg" width="200" /></td>
</tr>
<tr>
	<td><a id="second">2</a></td>
	<td><img src="/content.jpg" width="200" /></td>
	<td><img src="/style2.jpg"  width="200" /></td>
	<td><img src="/outputs-2/combined0.jpg" width="200" /></td>
</tr>
<tr>
	<td><a id="third">3</a></td>
	<td><img src="/content.jpg"  width="200" /></td>
	<td><img src="/style3.jpg"  width="200" /></td>
	<td><img src="/outputs-3/combined1.jpg" width="200" /></td>
</tr>
</table>

## Hyper-Parameters Analysis

### Style loss
The Style weigth should be a moderate value like 7-12 if the style image is has the style in the same form as that in the <a href="#first">first example</a>, Where the both were building but the style had to be partially copied.
<br/>
The weigth should be pretty low like 3-5 if the style image is only a style template on its on rather that its implemented images.
<br/>
The weigth should be very high like 25-30 if the style is not very distinctive in the image like <a href="#second">second</a> which is just a image of a modern city which I wanted to encapsulate into the old building.

### Content loss
This one is very simple to understand and has only one hyperparameter, the content weigth which can be put anything below 0.03 as this is the last thing we would want to optimize the image for . This will be automatically be done as the loss relation is linear which would make the algorithm to atleast have its main distinctive features like the outline, the base color and so on.
