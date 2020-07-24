# Style-Transfer
This is a piece of code which optimizes a given( if not given then random) image into something which takes the content from one image and style of another.
<br/>
I have used just the Tensorflow Keras backend so that the computations could be done with the help of multiple threads (or GPU) and used scipy's **fmin_l_bfgs_b** which just implements an advanced version of newtons formula for roots. But rather here it finds roots for the derivative of the loss function which should corresponds to its minima (most probably).
<br/>
## Demo
I have tried keeping the same content to experiment with the hyperparameters of the style loss.
<table>
<tr>
	<th>Content</th>
	<th>Style</th>
	<th>Output</th>
</tr>
<tr>
	<td><img src="/content.jpg"/></td>
	<td><img src="/style.jpg"/></td>
	<td></td>
</tr>
<tr>
	<td><img src="/content.jpg"/></td>
	<td><img src="/style2.jpg"/></td>
	<td></td>
</tr>
<tr>
	<td><img src="/content.jpg"/></td>
	<td><img src="/style3.jpg"/></td>
	<td></td>
</tr>
</table>

## Hyper-Parameters Analysis

### Style loss
The Style weigth should be a moderate value like 7-12 if the style image is 