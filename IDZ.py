import os
import cv2
import imutils
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from matplotlib import pyplot as plt
import scipy.ndimage
import math
import os
import csv


def readImage(filepath,num,ext):
	print("Reading input Image")
	srcImage =cv2.imread(filepath + '%s' %num + '.'+ ext)
	#srcImage =cv2.imread(filepath + 'i_'+'%s' %num + '.'+ ext)
	
	return srcImage
	
def imageProcessing(inputimg,destFolder):
	print("Image Processing Algorithm : Started")
	source=inputimg.copy()
	lab_image=cv2.cvtColor(inputimg,cv2.COLOR_BGR2LAB)
	l_channel,a_channel,b_channel=cv2.split(lab_image)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(l_channel)
	img2=cl1.astype(np.uint8)
	imgdummy=img2.copy()
	ret,thresh = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#new
	newimg = cv2.bitwise_not(thresh)#new
	
	#Flood Filled 
	im_floodfill = newimg.copy()
	h, w = newimg.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)
	cv2.floodFill(im_floodfill, mask, (0,0), 255);
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)


	##Combine the two images to get the foreground
	im_out = newimg | im_floodfill_inv
	
	#Pallor Extraction
	pallorImg=im_floodfill.copy()
	
	#Erode
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	eroded=cv2.erode(im_out,element)
	
	#Draw Contours 
	im2, contours, hierarchy = cv2.findContours(im_out,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(inputimg, contours, -1, (0,255,0), 3)
	cv2.imwrite(destFolder,inputimg)
	plt.figure(1)
	plt.imshow(inputimg)
	plt.show()
	print("Image Processing Algorithm : Completed")
	return contours
 
		

def contourFeatures(contours):
	print("Extraction of Contour Features")
	
	print("length of cell contours")
	print(len(contours))

	centroidX = []
	centroidY = []
	centroid=[]
	centroid1 =[]
	contourA=[]
	elong=[]
	buf=[]
	ar=[]
	peri=[]
	moment=[]
	convexhullArea=[]
	contourapprox=[]
	
	aspectRatioSum=0
	z=0
	elongSum=0
	perimeterSum=0
	contourAreaSum=0
	convexhullAreaSum=0
	contourapproxSum=0
	
	for c in contours:
		
		#Image Moments
		
		#M=cv2.moments(c.astype(np.float64))
		M=cv2.moments(c)
		moment.append(M)
		
		# Compute the center of the contour
		if M["m00"] != 0:
			cX = float(float(M["m10"]) / float(M["m00"]))
			cY = float(float(M["m01"]) /float(M["m00"]))
		else: 
			cX,cY=0,0
		centroid.append((float(cX),float(cY)))
		centroidX.append(cX)
		centroidY.append(cY)

		#cv2.drawContours(inputimg, contours, -1, (0,255,0), 3)
		#cv2.circle(inputimg, (cX, cY), 1, (255, 255, 255), -1)

		#contour area
		area=cv2.contourArea(c)
		contourA.append(area)
		contourAreaSum=contourAreaSum+area
		
		#elongation
		Ex = M['mu20'] + M['mu02']
		Ey = 4*M['mu11']**2 + (M['mu20'] - M['mu02'])**2
		if (Ex - Ey**0.5)!= 0: 
			elongation= (Ex + Ey**0.5) / (Ex - Ey**0.5)
		else:
			elongation=0
		elong.append(elongation)
		elongSum=elongSum+elongation

		#contour perimeter
		perimeter = cv2.arcLength(c,True)
		peri.append(perimeter)
		perimeterSum=perimeterSum+perimeter
		
		if  area < 15000:
			buf.append(c)
			
		# convex hull
		convex_hull = cv2.convexHull(c)
		
		# convex hull area
		convex_area = cv2.contourArea(convex_hull)
		convexhullArea.append(convex_area)
		convexhullAreaSum=convexhullAreaSum+convex_area
		
		#Bounding Rectangle
		x,y,w,h = cv2.boundingRect(c)
		
		#Aspect Ratio
		aspect_ratio = float(w)/h
		ar.append(aspect_ratio)
		aspectRatioSum=aspectRatioSum+aspect_ratio
		
		# solidity = contour area / convex hull area
		#solidity = area/float(convex_area)
		
		'''
		ellipse = cv2.fitEllipse(c)
		cv2.ellipse(displayframe,elps,(0,0,255))
		cv2.imshow("Perfectly fitted ellipses", displayframe)

		# center, axis_length and orientation of ellipse
		(center,axes,orientation) = ellipse
		
		majoraxis_length = max(axes)
		minoraxis_length = min(axes)
	
		# eccentricity = sqrt( 1 - (ma/MA)^2) --- ma= minor axis --- MA= major axis
		eccentricity = math.sqrt(1-(minoraxis_length/majoraxis_length)**2)
		'''
		
		# contour approximation
		#approx = cv2.approxPolyDP(c,0.02*perimeter,True)
		#contourapprox.append(approx)
		#contourapproxSum=contourapproxSum+approx
		
	return moment,centroid,contourA,contourAreaSum,elong,elongSum,peri,perimeterSum,convexhullArea,convexhullAreaSum,ar,aspectRatioSum


# Define window names
win_delaunay = "Delaunay Triangulation"
win_voronoi = "Voronoi Diagram"

#Turn on animation while drawing triangles
animate = True

#Define colors for drawing.
delaunay_color = (255,255,255)
delaunay_color = (0,0,0)
points_color = (0, 0, 255)



# Check if a point is inside a rectangle
def rect_contains(rect, point) :
	if point[0] < rect[0] :
		return False
	elif point[1] < rect[1] :
		return False
	elif point[0] > rect[2] :
		return False
	elif point[1] > rect[3] :
		return False
	return True

def draw_delaunay(img, subdiv, delaunay_color):
	
	dist=[]
	triangleList = subdiv.getTriangleList()
	size = img.shape
	r = (0, 0, size[1], size[0])
	
	for t in triangleList :
		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])

		if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
			#cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.CV_AA, 0)
			#cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.CV_AA, 0)
			#cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.CV_AA, 0)
			
			cv2.line(img, pt1, pt2, delaunay_color, 1, 0)
			cv2.line(img, pt2, pt3, delaunay_color, 1,  0)
			cv2.line(img, pt3, pt1, delaunay_color, 1,  0)
			
			dx1=(pt2[0]-pt1[0])*(pt2[0]-pt1[0])
			dy1=(pt2[1]-pt1[1])*(pt2[1]-pt1[1])
			d1=math.sqrt(abs((dx1+dy1)))
			dx2=(pt3[0]-pt2[0])*(pt3[0]-pt2[0])
			dy2=(pt3[1]-pt2[1])*(pt3[1]-pt2[1])
			d2=math.sqrt(abs((dx2+dy2)))
			dx3=(pt1[0]-pt3[0])*(pt1[0]-pt3[0])
			dy3=(pt1[1]-pt3[1])*(pt1[1]-pt3[1])
			d3=math.sqrt(abs((dx3+dy3)))

			dist.append(d1)
			dist.append(d2)
			dist.append(d3)
	
	return dist


def draw_voronoi(img, subdiv) :

	(facets, centers) = subdiv.getVoronoiFacetList([])
	
	for i in xrange(0,len(facets)) :
		ifacet_arr = []
		for f in facets[i] :
			ifacet_arr.append(f)
	
		ifacet = np.array(ifacet_arr, np.int)
		color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
 
		cv2.fillConvexPoly(img, ifacet, color, cv2.CV_AA, 0);
		ifacets = np.array([ifacet])
		cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.CV_AA, 0)
		cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.cv.CV_FILLED, cv2.CV_AA, 0)


def triangulationInitialization(inputimg,contours,centroid,denaulay):

	m=0
	
	size=inputimg.shape
	rect=(0,0,size[1], size[0])
	subdiv=cv2.Subdiv2D(rect)
	
	for p in centroid:
		m=m+1
		subdiv.insert(p)
		
		'''
		if animate :
				img_copy = inputimg.copy()
				# Draw delaunay triangles
				draw_delaunay(img_copy, subdiv,(10, 10, 10))
				cv2.imshow(win_delaunay, img_copy)
				cv2.waitKey(1000)
		'''
		
	distance=draw_delaunay(inputimg, subdiv,(0, 0, 0))
	
	#s1,s2,s3=draw_delaunay(inputimg, subdiv,(0, 0, 0))
	#draw_delaunay(inputimg, subdiv,(0, 0, 0))

	
	cv2.imshow(win_delaunay,inputimg)
	cv2.waitKey(10000)
	#cv2.imwrite('C:/Users/usb7kor/Desktop/test/vtaMur.jpg',inputimg)
	cv2.imwrite(denaulay,inputimg)
	return distance

def deriveFeatures(distance,moment,centroid,contourA,contourAreaSum,elong,elongSum,peri,perimeterSum,convexhullArea,convexhullAreaSum,ar,aspectRatioSum):
	print("derivedFeatures")
	
	
	distancediff=[]
	
	sum_diffdist=0
	sumDist=0
	contourAreaAvg=0
	elongAvg=0
	maxcontArea=0
	maxElong=0
	maxPerimeter=0
	perimeterAvg=0
	maxHullArea=0
	hullAreaAvg=0
	maxAspectRatio=0
	aspectRatioAvg=0
	maxApprox=0
	contourApproxAvg=0
	contourLength=0
	
	contourLength=len(contours)
	
	for k in distance:
		sumDist=sumDist+k	
	avgDist=sumDist/len(distance)

	for l in distance:
		distdiff=l-avgDist
		distdiff=distdiff*distdiff
		distancediff.append(distdiff)
		sum_diffdist=sum_diffdist+distdiff
	
	Variance=sum_diffdist/len(distance)
	#SD=math.sqrt(Variance);
	SD1=sum_diffdist/(len(distance)-1)
	SD=math.sqrt(SD1)
	CV=(SD/avgDist)*100
	
	#ContourArea
	maxcontArea=max(contourA)
	contourAreaAvg=contourAreaSum/len(contours)
	
	#Elongation
	maxElong=max(elong)
	elongAvg=elongSum/len(contours)
	
	#Perimeter
	maxPerimeter=max(peri)
	perimeterAvg=perimeterSum/len(contours)
	
	#ConvexHullArea
	maxHullArea=max(convexhullArea)
	hullAreaAvg=convexhullAreaSum/len(contours)
	
	#Aspect Ratio
	maxAspectRatio=max(ar)
	aspectRatioAvg=aspectRatioSum/len(contours)
	
	#Contour Approximation
	#maxApprox=max(contourapprox)
	#contourApproxAvg=contourapproxSum/len(contours)
	
	#printResults(avgDist,Variance,SD1,SD,CV,maxcontArea,contourAreaAvg,maxElong,elongAvg,maxPerimeter,perimeterAvg,maxHullArea,hullAreaAvg,maxAspectRatio,aspectRatioAvg,distance,contourLength)
	print("len(distance)")
	print(len(distance))
	
	return avgDist,Variance,SD1,SD,CV,maxcontArea,contourAreaAvg,maxElong,elongAvg,maxPerimeter,perimeterAvg,maxHullArea,hullAreaAvg,maxAspectRatio,aspectRatioAvg,distance,contourLength
	
	
def printResults(avgDist,Variance,SD1,SD,CV,maxcontArea,contourAreaAvg,maxElong,elongAvg,maxPerimeter,perimeterAvg,maxHullArea,hullAreaAvg,maxAspectRatio,aspectRatioAvg,distance,contourLength):
	
	print("avgDist:",avgDist)
	print("Variance:",Variance)
	print("SD1:",SD1)
	print("SD:",SD)
	print("CV:",CV)	
	print("maxcontArea:",maxcontArea)
	print("Variance:",contourAreaAvg)
	print("SD1:",maxElong)
	print("SD:",elongAvg)
	print("CV:",maxPerimeter)	
	print("Aspect_Ratio_Avg:",perimeterAvg)
	print("Perimeter:",maxHullArea)
	print("Maximum contour area:",hullAreaAvg)
	print("maxAspectRatio:",maxAspectRatio)
	print("aspectRatioAvg:",aspectRatioAvg)
	print("distance:",distance)	
	print("contourLength:",contourLength)	
	
	
	
def filesave(data,fname):
	path= "%s%s.%s" % ("C:\\Users\\usb7kor\\Desktop\\IDZ\\",fname,"csv")
	data=str(data).replace('[','').replace(']','')
	with open(path, 'w') as output:
		output.write(str(data))   	




### Main function ###

filepath = "C:\\Users\\usb7kor\\Desktop\\1\\"

ext='jpg'
imageCount=1
os.chdir(filepath)
buf=[]
for num, filename in enumerate(os.listdir(os.getcwd()), start= 1):
	if (num<=imageCount):
		destFolder=filepath + '%s' %num + '_labelled' + '.' + ext
		denaulay=filepath + '%s' %num + '_Denaulay' + '.' + ext
		fname = filename
		srcImage=readImage(filepath,num,ext)
		contours=imageProcessing(srcImage,destFolder)
		moment,centroid,contourA,contourAreaSum,elong,elongSum,peri,perimeterSum,convexhullArea,convexhullAreaSum,ar,aspectRatioSum=contourFeatures(contours)
		distance=triangulationInitialization(srcImage,contours,centroid,denaulay)
		avgDist,Variance,SD1,SD,CV,maxcontArea,contourAreaAvg,maxElong,elongAvg,maxPerimeter,perimeterAvg,maxHullArea,hullAreaAvg,maxAspectRatio,aspectRatioAvg,distance,contourLength=deriveFeatures(distance,moment,centroid,contourA,contourAreaSum,elong,elongSum,peri,perimeterSum,convexhullArea,convexhullAreaSum,ar,aspectRatioSum)
		
		buf.append('contourLength')
		buf.append('avgDist')
		buf.append('Variance')
		buf.append('SD1')
		buf.append('SD')
		buf.append('CV')
		buf.append('maxcontArea')
		buf.append('contourAreaAvg')
		buf.append('maxElong')
		buf.append('elongAvg')
		buf.append('maxPerimeter')
		buf.append('perimeterAvg')
		buf.append('maxHullArea')
		buf.append('hullAreaAvg')
		buf.append('maxAspectRatio')
		buf.append('aspectRatioAvg')
		
		
		fname1= "%s_%s" % ("metrics",num)
		filesave(buf, fname1)
		
		path= "%s%s.%s" % ("C:\\Users\\usb7kor\\Desktop\\IDZ\\",fname1,"csv")
		buf1=[]
		buf1.append(contourLength)
		buf1.append(avgDist)
		buf1.append(Variance)
		buf1.append(SD1)
		buf1.append(SD)
		buf1.append(CV)
		buf1.append(maxcontArea)
		buf1.append(contourAreaAvg)
		buf1.append(maxElong)
		buf1.append(elongAvg)
		buf1.append(maxPerimeter)
		buf1.append(perimeterAvg)
		buf1.append(maxHullArea)
		buf1.append(hullAreaAvg)
		buf1.append(maxAspectRatio)
		buf1.append(aspectRatioAvg)
		
		
		
		with open(path,'a',newline='') as f:
			writer=csv.writer(f)
			writer.writerow([])
			writer.writerow(buf1)
		fname= "%s_%s" % ("distance",num)
		filesave(distance, fname)

	
print("Process Completed")
cv2.destroyAllWindows()







