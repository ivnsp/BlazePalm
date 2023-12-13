import os
import torch
import cv2 as cv
import numpy as np

from blazepalm import PalmDetector
from handlandmarks import HandLandmarks

m = PalmDetector()
m.load_weights("./palmdetector.pth")
m.load_anchors("./anchors.npy")

hl = HandLandmarks()
hl.load_weights("./handlandmarks.pth")


def inference(frame):
	#frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	hh, ww, _ = frame.shape
	ll = min(hh, ww)
	img = cv.resize(frame[:ll, :ll][:, ::-1], (256, 256))
	predictions = m.predict_on_image(img)
	for pred in predictions:
		for pp in pred:
			print(pp)
			p = pp[:-1] * ll
			score = pp[-1].item()
			p = p.numpy().astype(int)
			# p contains 18 values
			# [xmin, ymin, xmax, ymax, 7 xy coordinates of landmarks]
			cv.rectangle(frame, (p[0], p[1]), (p[2], p[3]), (255, 0, 0), 2)
			for i in range(7):
				cv.circle(frame, (p[4+i*2], p[4+i*2+1]), 2, (0, 0, 255), 2)
			cv.putText(frame, str(score), (p[0], p[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
			
			# skip landmarks for now as they are not working
			""" # crop this image, pad it, run landmarks
			x = max(0, p[0].type(torch.int))
			y = max(0, p[1].type(torch.int))
			endx = min(ll, p[2].type(torch.int))
			endy = min(ll, p[3].type(torch.int))
			cropped_hand = frame[y:endy, x:endx]	
			maxl = max(cropped_hand.shape[0], cropped_hand.shape[1])
			#cropped_hand = np.pad(cropped_hand, 
			#			(((maxl-cropped_hand.shape[0])//2, (maxl-cropped_hand.shape[0]+1)//2), ((maxl-cropped_hand.shape[1])//2, (maxl-cropped_hand.shape[1]+1)//2)),
			#			'constant')
			cropped_hand = cv.resize(cropped_hand, (256, 256))
			_, _, landmarks = hl(torch.from_numpy(cropped_hand).permute((2, 0, 1)).unsqueeze(0))
			print(landmarks) """
			
	cv.imshow('frame', frame)
	

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description='BlazePalm static inference')
	parser.add_argument('-i', '--input', help='path to input image dir')
	parser.add_argument('-o', '--output', help='path to output image dir', default='.')
	parser.add_argument('-s', '--save', help='save output images', default=False)
	args = parser.parse_args()

	for i, imgpath in enumerate(os.listdir(args.input)):
		img = cv.imread(os.path.join(args.input, imgpath))
		inference(img)
		if args.save:
			cv.imwrite(os.path.join(args.output, imgpath), img)
		cv.waitKey(0)
		cv.destroyAllWindows()