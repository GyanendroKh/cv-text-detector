from imutils.object_detection import non_max_suppression
from skimage.metrics import structural_similarity
import numpy as np
import tqdm
import imutils
import time
import click
import cv2
import os
import math


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


def compute_regions_to_ignore(hog, img):
	(regions, _) = hog.detectMultiScale(
		img,
		winStride=(4, 4),
		padding=(8, 8),
		scale=1.05
	)

	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in regions])
	pick = non_max_suppression(rects, probs=0.5, overlapThresh=0.65)

	region_to_ignore = []

	for (startX, startY, endX, endY) in pick:
		startX = int(startX)
		startY = int(startY)
		endX = int(endX)
		endY = int(endY)

		region_to_ignore.append(((startX, startY), (endX, endY)))
	return region_to_ignore


def detect_diff(img1, img2):
	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	(score, _) = structural_similarity(img1, img2, full=True)
	return score

def ignore_region(actual_region):
	(startX, startY, endX, endY) = actual_region

	if (endX < 85 and endY < 32) or (startX > 350 and startY > 30):
		return True

	return False


def run_detection(east, video, width, height, output, diff, no_draw):
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	net = cv2.dnn.readNet(east)
	vs = cv2.VideoCapture(video)

	fps = vs.get(cv2.CAP_PROP_FPS)
	frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
	prev = None

	i = 0
	for j in tqdm.trange(int(frame_count/fps)):
		frame = vs.read()
		vs.set(cv2.CAP_PROP_POS_MSEC, (i*1000))
		i += 1
		frame = frame[1]

		if frame is None:
			break

		frame = cv2.resize(frame, (width, height))

		orig = frame.copy()

		if prev is None:
			prev = frame
		else:
			d = detect_diff(frame, prev)
			prev = frame
			if d > diff:
				continue

		blob = cv2.dnn.blobFromImage(frame, 1.0, (width, height),
			(123.68, 116.78, 103.94), swapRB=True, crop=False)
		net.setInput(blob)
		(scores, geometry) = net.forward(layerNames)

		(rects, confidences) = decode_predictions(scores, geometry)
		boxes = non_max_suppression(np.array(rects), probs=confidences)

		regions = []

		for (startX, startY, endX, endY) in boxes:
			startX = int(startX)
			startY = int(startY)
			endX = int(endX)
			endY = int(endY)

			if ignore_region((startX, startY, endX, endY)):
				continue

			regions.append(((startX, startY), (endX, endY)))

		if len(regions) <= 4:
			continue

		if not no_draw:
			for ((startX, startY), (endX, endY)) in regions:
				cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

		cv2.imwrite(os.path.join(output, f'frame-{i}.jpg'), orig)

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
@click.option('--output',
	default='./output', type=click.Path(),
	help='Output folder for the images (frame).'
)
@click.option('--diff',
	default=0.80, type=float,
	help='Probability for image difference (1 for excat match).'
)
@click.option('--no-draw', is_flag=True, help='Should skip drawing boxes.')
def main(east, video, output, diff, no_draw):
	if os.path.exists(output):
		if not os.path.isdir(output):
			os.mkdir(output)
	else:
		os.mkdir(output)
	run_detection(east, video, 480, 320, output, diff, no_draw)


if __name__ == "__main__":
	main()
