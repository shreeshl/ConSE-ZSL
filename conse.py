from __future__ import division
import numpy as np
import caffe
import os
import sys
import re
import cPickle as pickle
from sklearn.metrics.pairwise import euclidean_distances
from progressbar import  Percentage, ProgressBar, Bar
import datetimea

def get_avearage_vector(word_list,n):
	temp = np.zeros((1,n))[0]
	count = 0
	for item in word_list:
		if item in word_vectors:
			temp = temp+word_vectors[item]
		count+=1
	return temp/count


def load_word_vectors(file, directory = ''):
	os.chdir(directory)
	word_vectors = {}
	f = open(file,'r')
	g = f.read()
	g = g.split('\n')[:-1]
	count = 0
	pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(g)).start()
	for item in g:
		pbar.update(count)
		count+=1
		index = item.find(' ')
		word_vectors[ item[:index] ] = [float(number) for number in item[index+1:].split(' ')]
	pbar.finish()
	return word_vectors


def process_labels(labels):
	processed_labels = []
	for item in labels:
		temp = re.findall(r'[\w-]+',item.lower(), re.IGNORECASE)[1:]
		temp = list(set(temp))
		processed_labels.append(temp)
	return np.array(processed_labels)



caffe.set_mode_cpu()

#caffe.set_device(0)  
#caffe.set_mode_gpu()
T = input('T (Top T probabilities): \n')
n = input('n (Dimension of Word Vector) : \n')
caffe_root = '/Users/shreesh/Downloads/caffe-master/'
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',caffe.TEST)



transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) 
transformer.set_raw_scale('data', 255)  
transformer.set_channel_swap('data', (2,1,0))


labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')

labels_without_synsets = process_labels(labels)
wnid_dict = pickle.load( open( "/Users/shreesh/Academics/CS772/ZSL/ConSE/dictionary.p", "rb" ))


word_vectors = load_word_vectors('glove.6B.50d.txt', '/Users/shreesh/Academics/CS772/ZSL/ConSE/glove.6B')

filenames = []
test_classes = []
test_classes_wnid = []
embedding = []
i = 0
count = 0

os.chdir('/Users/shreesh/Academics/CS772/ZSL/ConSE/2hop')
pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(os.listdir(os.getcwd()))).start()
for filename in os.listdir(os.getcwd()):
	pbar.update(count)
	if filename=='synset_n04134632.jpg' : continue
	if 'DS' in filename: continue

	filenames.append(filename)
	net.blobs['data'].data[i] = transformer.preprocess('data', caffe.io.load_image(filename))
	test_classes.append( wnid_dict[filename.split('_')[1][:-4]] )
	test_classes_wnid.append(filename.split('_')[1][:-4])
	i+=i
	count+=1
	
	if i==50:
		
		out = net.forward()
		for k in range(i):
			probs = out['prob'][k].copy()
			embedding = get_embedding(probs)
			embedding.append(get_embedding(probs))
			
		i = i%50
		

for k in range(i):
	probs = out['prob'][k].copy()
	embedding.append(get_embedding(probs))


def cosine_similarity(vecx, vecy):
    norm = np.sqrt(np.dot(vecx, vecx))* np.sqrt(np.dot(vecy, vecy))
    return np.dot(vecx, vecy) / (norm + 1e-10)
    #return np.dot(vecx, vecy) / (norm)

def get_dist(label, item):
	label = re.findall(r'[\w-]+',label.lower(), re.IGNORECASE)
	avearage_label = get_avearage_vector(label,n)
	#return euclidean_distances(item, avearage_label)
	return cosine_similarity(item, avearage_label)



def get_accuracy(embedding, test_classes, test_classes_wnid):
	accuracy_1,accuracy_2, accuracy_5, accuracy_10 = 0,0,0,0
	
	pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(embedding)).start()
	
	start = datetime.datetime.now()
	
	for m,item in enumerate(embedding):
		pbar.update(m)
		closest_list = []
		
		for label_list in test_classes:
			dist = get_dist(label_list,item)
			closest_list.append(dist)
		
		closest_list = np.array(closest_list)
		closest = closest_list.argsort()[::-1][:10]
		
		actual = wnid_dict[test_classes_wnid[m]] 
		for i,index in enumerate(closest):
			if actual==test_classes[index]:
				if i==0 : 
					accuracy_1+=1
					accuracy_2+=1
					accuracy_5+=1
					accuracy_10+=1
					break
				if i>0 and i<=1 : 
					accuracy_2+=1
					accuracy_5+=1
					accuracy_10+=1
					break
				if i>1 and i<=4 : 
					accuracy_5+=1
					accuracy_10+=1
					break
				if i>4 and i<=9 : 
					accuracy_10+=1
					break
			"""if test_classes[index]==actual:
				accuracy_5+=1
				break"""
	pbar.finish()
	print datetime.datetime.now() - start
	return accuracy_1, accuracy_2, accuracy_5, accuracy_10


accuracy_1,accuracy_2, accuracy_5, accuracy_10 = get_accuracy(embedding, test_classes, test_classes_wnid)
accuracy_1 = 100*accuracy_1/len(filenames)
accuracy_5 = 100*accuracy_5/len(filenames)
print 'Hit@1 : %f'%(accuracy_1) + '%'
print 'Hit@5 : %f'%(accuracy_5) + '%'

################################################################
################################################################
################################################################


probs = np.fromfile('feature.bin',dtype = np.float32)
probs = probs.reshape((2000,-1))

def remove_underscore(i):
	return i.split('_')[0]

test_classes_wnid_2 = map(remove_underscore,open('id.txt','r').read().strip().split())
pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(probs)).start()

word_vectors = load_word_vectors('glove.6B.100d.txt', '/Users/shreesh/Academics/CS772/ZSL/ConSE/glove.6B')
embedding_2 = []
test_classes_2 = []

def get_embedding(probs):
	indices = probs.argsort()[::-1][:T]
	normalization_constant = probs[indices].sum()
	temp = np.zeros((1,n))[0]
	for m in range(T):
		temp += probs[indices][m]*get_avearage_vector(labels_without_synsets[indices][m],n)
	temp = temp/normalization_constant
	return temp


for count, prob in enumerate(probs):
	pbar.update(count)
	test_classes_2.append(wnid_dict[synsets[count]])
	embedding_2.append(get_embedding(prob))


accuracy_11,accuracy_22, accuracy_55, accuracy_1010 = get_accuracy(embedding_2, test_classes_2, test_classes_wnid_2)
accuracy_11 = 100*accuracy_11/len(embedding_2)
accuracy_55 = 100*accuracy_55/len(embedding_2)
print 'Hit@1 : %f'%(accuracy_11) + '%'
print 'Hit@5 : %f'%(accuracy_55) + '%'






