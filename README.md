Polygon Art Generation
======================
Generates low-polygon art from existing images, using Voronoi diagrams and the 2D discrete cosine transform. Uses Python3, NumPy, SciPy, scikit-image, and the Tkinter GUI library.

Usage
-----
To use, run `python3 polygon-art-generation.py`. Select a 24-bit uncompressed .bmp file from the open file prompt. Then enter an integer to specify the number of polygons. After a few minutes, the low polygon image will be displayed alongside the original image. It will also save the generated image as a new bitmap image, in the same folder as the input file.

Implementation
--------------
The program has three main components, used for drawing polygons, determining the color of each polygon, and determining the distribution of polygons.

### Polygon creation
The polygons must be of variable sizes and at variable coordinates, as different areas of the image should be represented with different levels of detail. The concept of Voronoi diagrams serves this purpose. Consider a 2D plane, and plot a series of points (called Voronoi points) randomly throughout this plane. For each Voronoi point x, its Voronoi region is defined as the set of all points in the plane for which x is the nearest of all Voronoi points. By placing a finite number of Voronoi points in a plane, the plane is partitioned into an equal number of Voronoi regions, each of which is a polygon. The diagram below shows a Voronoi diagram consisting of 20 randomly placed Voronoi points, with a random color assigned to each Voronoi region. Selecting any random point in the image, the closest black dot (Voronoi point) to that point will be the Voronoi point located in the same color region as the random point.

<img src="https://user-images.githubusercontent.com/31748813/100980647-68d21980-34fa-11eb-8d60-1955f54fd546.png" height="300px" />

By creating a Voronoi diagram for a number of Voronoi points at selected locations, and choosing appropriate colors for each Voronoi region, a low-polygon image will be generated. If there are N Voronoi points used to create an image, then there will be N polygons in the generated image. To draw the Voronoi diagrams, a KD-tree is used, so that the closest Voronoi point for any pixel can be computed efficiently without having to compute its distance to every Voronoi point.

### Color selection
Color selection is important, as the polygons should approximate the colors of the original image, and the contents of the low-poly image should still be recognizable. Given the region of a polygon, and all pixels within that region in the original image, a single color must be created for that polygon in the low-poly image. The Lab color space (also called CIE L\*a\*b\* color) is based on human perception and designed so that the distance between two colors in this color space corresponds to a roughly equal amount of visual change when perceived by people. Therefore, averaging the colors using this color space (by converting from RGB to Lab, computing the average color, then converting back to RGB) produces a good representative color for the region of each polygon.

### Polygon distribution
Because the polygons are drawn using Voronoi diagrams, the polygon distribution is determined by the locations of the Voronoi points. The polygon distribution method must also generate a high density of small polygons in image regions with significant detail and color variation, such as the foreground subject of the image, and a low density of large polygons in image regions with relatively uniform color, such as in the image's background; this requires determining the degree of pixel change within different areas of the image. The 2D discrete cosine transform is well-suited for this problem. The 2D DCT transforms an image signal into its various low and high frequency components. In image compression, DCT is used to identify the different frequency components of an image signal, so that the higher frequency components can be either discarded or significantly quantized and compressed, while preserving the low frequency components. Here, however, DCT is used for a somewhat inverse purpose, discarding most of the low frequency components. The high frequency components of an image signal are directly related to the pixel changes that occurs within that image. If an image has a lot of color variations, then the high frequency coefficients computed by the 2D DCT will have large magnitudes. In this algorithm, however, the 2D DCT is not performed on the RGB channels of the image. Instead, the image is converted to YUV, and the 2D DCT is only applied to the Y luminance channel. Although this means that the high-frequency components correspond to changes in brightness, this method functions well in detecting pixel variation in the original image and identifying high-detail regions, and performs better than using the RGB data.

Voronoi points are thus selected as follows. First, the 2D DCT is applied to 8x8 blocks within the image. Next, each block has its entries divided by a quantization matrix that reduces the magnitude of very low frequency components drastically, preserves the magnitude of very high frequency components, and applies a reduction somewhere in between for medium frequency components. The magnitudes of these new component values are then summed to produce a ‘detail’ score for each block in the image. To generate a Voronoi point, a random block is chosen, where the probability of choosing each block is directly proportional to its ‘detail’ score; a Voronoi point is then placed uniformly at random within that block, and this process is repeated until N Voronoi points have been selected. This has the effect that blocks in the image with high levels of detail will be more likely to receive Voronoi points, which creates a higher density of polygons in high-detail regions.

Sample images
-------------
| Original image | Low-polygon image |
|-|-|
|![nebula](https://user-images.githubusercontent.com/31748813/101319658-2fa9ea00-3817-11eb-95c8-898d92cc8759.jpg) <br/>Nebula|![nebula_lowpoly_750](https://user-images.githubusercontent.com/31748813/101319697-3cc6d900-3817-11eb-9479-35d3eab44ed1.jpg) <br/>N=750|
|![eye](https://user-images.githubusercontent.com/31748813/101319792-6122b580-3817-11eb-9a70-7dec76a2c1fb.jpg) <br/>Human eye|![eye_lowpoly_700](https://user-images.githubusercontent.com/31748813/101319828-7566b280-3817-11eb-8bb9-43af063a90c0.jpg) <br/>N=700|
|![rose](https://user-images.githubusercontent.com/31748813/101319848-7b5c9380-3817-11eb-8ade-65f6b8e75e14.jpg) <br/>Rose|![rose_lowpoly_300](https://user-images.githubusercontent.com/31748813/101319851-7c8dc080-3817-11eb-95ae-a3de3e5ae568.jpg) <br/>N=300|
|![giraffe](https://user-images.githubusercontent.com/31748813/101319852-7e578400-3817-11eb-84ad-7e6bfeca4020.jpg) <br/>Giraffe|![giraffe_lowpoly_200](https://user-images.githubusercontent.com/31748813/101319858-831c3800-3817-11eb-9d9d-1e6e5fee6c6a.jpg) <br/>N=200|
||![giraffe_lowpoly_650](https://user-images.githubusercontent.com/31748813/101319861-84e5fb80-3817-11eb-9ab8-bfc3cb6768b7.jpg) <br/>N=650|
||![giraffe_lowpoly_900](https://user-images.githubusercontent.com/31748813/101319865-86172880-3817-11eb-8e28-b7c6522f0892.jpg) <br/>N=900|
|![helmet](https://user-images.githubusercontent.com/31748813/101319868-87e0ec00-3817-11eb-88b6-09024c53d8b3.jpg) <br/>Helmet|![helmet_lowpoly_1000](https://user-images.githubusercontent.com/31748813/101319873-89aaaf80-3817-11eb-8dd0-f6ce302fbed6.jpg) <br/>N=1000|
|![wave](https://user-images.githubusercontent.com/31748813/101319876-8b747300-3817-11eb-9831-257b722b89c9.jpg) <br/>Wave|![wave_lowpoly_1000](https://user-images.githubusercontent.com/31748813/101319881-8ca5a000-3817-11eb-9b62-ffc21b5ef543.jpg) <br/>N=1000|

