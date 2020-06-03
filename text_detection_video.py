from imutils.object_detection import non_max_suppression
import numpy as np
import tqdm
import imutils
import time
import click
import cv2
import math


def format_time(t):
	hrs = math.floor(t / 3600)
	mins = math.floor((t % 3600) / 60)
	secs = math.floor(t % 60)

	t = ''
	
	if hrs > 0:
		if hrs < 10:
			t += '0'
		else:
			t += ''
	
	if mins < 10:
		t += f'{hrs}_0'
	else:
		t += f'{hrs}_'

	if secs < 10:
		t += f'{mins}_0'
	else:
		t += f'{mins}_'
	
	t += f'{secs}'

	return t


def decode_predictions(scores, geometry):
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		for x in range(0, numCols):
			if scoresData[x] < 0.5:
				continue

			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	return (rects, confidences)


def run_detection(east, video, time_for_frame, width, height):
	(W, H) = (None, None)
	(newW, newH) = (width, height)
	(rW, rH) = (None, None)

	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	print("[INFO] Loading EAST text detector...", end='')
	net = cv2.dnn.readNet(east)
	print("\r[INFO] Loaded EAST text detector...")

	hog = cv2.HOGDescriptor() 
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 

	vs = cv2.VideoCapture(video)

	fps = vs.get(cv2.CAP_PROP_FPS)
	frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
	duration = frame_count/fps

	i = 0
	for j in tqdm.trange(int(duration//(time_for_frame/1000))):
		frame = vs.read()
		vs.set(cv2.CAP_PROP_POS_MSEC, (i*time_for_frame))
		i += 1

		frame = frame[1]

		if frame is None:
			break
		
		frame = imutils.resize(frame, width=min(400, frame.shape[1]))
		orig = frame.copy()

		if W is None or H is None:
			(H, W) = frame.shape[:2]
			rW = W / float(newW)
			rH = H / float(newH)

		frame = cv2.resize(frame, (newW, newH))

		(h_regions, _) = hog.detectMultiScale(
			frame,
			winStride=(4, 4),
			padding=(8, 8),
			scale=1.05
		)

		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in h_regions])
		pick = non_max_suppression(rects, probs=0.5, overlapThresh=0.65)

		region_to_ignore = []

		for (startX, startY, endX, endY) in pick:
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)

			region_to_ignore.append(((startX, startY), (endX, endY)))

		blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
			(123.68, 116.78, 103.94), swapRB=True, crop=False)
		net.setInput(blob)
		(scores, geometry) = net.forward(layerNames)

		(rects, confidences) = decode_predictions(scores, geometry)
		boxes = non_max_suppression(np.array(rects), probs=confidences)

		regions = []

		for (startX, startY, endX, endY) in boxes:
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)

			ignore = False

			for ((iStartX, iStartY), (iEndX, iEndY)) in region_to_ignore:
				if startX > iStartX and endX < iEndX:
					ignore = True
			
			if ignore:
				continue

			regions.append(((startX, startY), (endX, endY)))

		if len(regions) == 0:
			continue
		
		for ((startX, startY), (endX, endY)) in region_to_ignore:
			cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 255), 2)

		for ((startX, startY), (endX, endY)) in regions:
			cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

		cv2.imwrite(f'frame-{format_time(j * int((time_for_frame/1000)))}.jpg', orig)

	vs.release()
	cv2.destroyAllWindows()


@click.command()
@click.option(
	'--east',
	default='frozen_east_text_detection.pb',
	type=click.Path(exists=True), show_default=True,
	help='Path to the East Text Detector Model.',
)
@click.option('--video',
	type=click.Path(exists=True),
	help='Path to the video file.'
)
@click.option('--duration',
	default=1500, type=int, show_default=True,
	help='Duration per frame (in milliseconds).',
)
def main(east, video, duration):
	run_detection(east, video, duration, 320, 320)

if __name__ == "__main__":
	main()
