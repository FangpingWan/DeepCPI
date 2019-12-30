from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from keras.models import load_model
import keras.backend as K
from keras.models import Sequential,Model
from keras.layers import merge
from keras.models import model_from_json
from keras.regularizers import l1,l2
from keras.layers.normalization import BatchNormalization
#from keras.utils.visualize_util import plot
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import Adagrad, Adam
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
import pickle 
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec
from gensim import corpora, models
import gensim


def compute_compound_feature(a_compound):
	"""
	Input:
		a_compound: InChI of the compound (string format)
	Output:
		if the compound is valid, return its 200-dimension feautre;
		else return string "error"
	"""
	m = Chem.MolFromInchi(a_compound)
	try:
		sentence1 = ''
		flag1 = 0
		for atom in xrange(m.GetNumAtoms()):
			info = {}
			fp = AllChem.GetMorganFingerprint(m, 2, useFeatures=False,fromAtoms=[atom],bitInfo=info)
			bits = list(fp.GetNonzeroElements())	
			for i in bits:
				position = info[i]
				if position[0][1] == 1:
					if flag1 == 0:
						flag1 = 1
						sentence1 += str(i)
					else:
						sentence1 += ' '
						sentence1 += str(i)
	except:
		return 'error'
	dictionary = corpora.Dictionary.load('dict_for_1_nofeatureinvariant2.dict')
	tfidf = models.tfidfmodel.TfidfModel.load('tfidf_for_1_nofeatureinvariant2.tfidf')
	lsi = models.lsimodel.LsiModel.load('lsi_for_1_nofeatureinvariant2.lsi')
	text = [word for word in sentence1.split()]
	from collections import defaultdict
	frequency = defaultdict(int)
	for token in text:
		frequency[token] += 1
	corpus = dictionary.doc2bow(text) 
	corpus_tfidf = tfidf[corpus]
	corpus_lsi = lsi[corpus_tfidf]
	lv = np.zeros(200)
	for i in xrange(len(corpus_lsi)):
		lv[corpus_lsi[i][0]] = corpus_lsi[i][1]
	return lv


def compute_compound_feature_(a_compound, dictionary, tfidf, lsi):
	"""
	Input:
		a_compound: InChI of the compound (string format)
		dictionary: pretrained dictionary for lsi
		tfidf: pretrained tfidf for lsi
		lsi: pretrained lsi
	Output:
		if the compound is valid, return its 200-dimension feautre;
		else return string "error"
	"""
	m = Chem.MolFromInchi(a_compound)
	try:
		sentence1 = ''
		flag1 = 0
		for atom in xrange(m.GetNumAtoms()):
			info = {}
			fp = AllChem.GetMorganFingerprint(m, 2, useFeatures=False,fromAtoms=[atom],bitInfo=info)
			bits = list(fp.GetNonzeroElements())	
			for i in bits:
				position = info[i]
				if position[0][1] == 1:
					if flag1 == 0:
						flag1 = 1
						sentence1 += str(i)
					else:
						sentence1 += ' '
						sentence1 += str(i)
	except:
		return 'error'
	#dictionary = corpora.Dictionary.load('dict_for_1_nofeatureinvariant2.dict')
	#tfidf = models.tfidfmodel.TfidfModel.load('tfidf_for_1_nofeatureinvariant2.tfidf')
	#lsi = models.lsimodel.LsiModel.load('lsi_for_1_nofeatureinvariant2.lsi')
	text = [word for word in sentence1.split()]
	from collections import defaultdict
	frequency = defaultdict(int)
	for token in text:
		frequency[token] += 1
	corpus = dictionary.doc2bow(text) 
	corpus_tfidf = tfidf[corpus]
	corpus_lsi = lsi[corpus_tfidf]
	lv = np.zeros(200)
	for i in xrange(len(corpus_lsi)):
		lv[corpus_lsi[i][0]] = corpus_lsi[i][1]
	return lv


def compute_protein_feature(a_protein):
	"""
	Input:
		a_protein: a protein sequence (string format)
	Output:
		return its 100-dimension feautre
	"""
	model = Word2Vec.load('new_word2vec_model')
	value = a_protein.lower()
	
	count1 = 0		
	features = np.zeros(100)

	begin = 0
	step = 3
	while True:
		if begin+step > len(value):
			break
		else:
			try:
				features += model[value[begin:begin+step]]
				begin += step
				count1 += 1
			except:
				begin += step
				continue

	begin = 1
	step = 3
	while True:
		if begin+step > len(value):
			break
		else:
			try:
				features += model[value[begin:begin+step]]
				begin += step
				count1 += 1
			except:
				begin += step
				continue

	begin = 2
	step = 3
	while True:
		if begin+step > len(value):
			break
		else:
			try:
				features += model[value[begin:begin+step]]
				begin += step
				count1 += 1
			except:
				begin += step
				continue

	features = features/float(count1)
	return features


def compute_protein_feature_(a_protein, model):
	"""
	Input:
		a_protein: a protein sequence (string format)
		model: pretrained word2vec model
	Output:
		return its 100-dimension feautre
	"""
	#model = Word2Vec.load('new_word2vec_model')
	value = a_protein.lower()
	
	count1 = 0		
	features = np.zeros(100)

	begin = 0
	step = 3
	while True:
		if begin+step > len(value):
			break
		else:
			try:
				features += model[value[begin:begin+step]]
				begin += step
				count1 += 1
			except:
				begin += step
				continue

	begin = 1
	step = 3
	while True:
		if begin+step > len(value):
			break
		else:
			try:
				features += model[value[begin:begin+step]]
				begin += step
				count1 += 1
			except:
				begin += step
				continue

	begin = 2
	step = 3
	while True:
		if begin+step > len(value):
			break
		else:
			try:
				features += model[value[begin:begin+step]]
				begin += step
				count1 += 1
			except:
				begin += step
				continue

	features = features/float(count1)
	return features


def load_data_with_label(filename):
	"""
	Input:
		tsv file with three columns (compound_inchi protein_sequence label)
	Output:
		return a list: [[InChI0,protein_sequence0,label0],[InChI1,protein_sequence1,label1],...]
	"""
	l = []
	f = open(filename)
	lines = f.readlines()
	f.close()
	for i in lines:
		parsed = i.strip('\n').strip('\r').split('\t')
		inchi = parsed[0]
		seq = parsed[1]
		label = parsed[2]
		l.append([inchi, seq, label])
	return l 



def load_data_without_label(filename):
	"""
	Input:
		tsv file with two columns (compound_inchi protein_sequence)
	Output:
		return a list: [[InChI0,protein_sequence0],[InChI1,protein_sequence1],...]
	"""
	l = []
	f = open(filename)
	lines = f.readlines()
	f.close()
	for i in lines:
		parsed = i.strip('\n').strip('\r').split('\t')
		inchi = parsed[0]
		seq = parsed[1]
		l.append([inchi, seq])
	return l 

def compute_feature_for_dataset(Data):
	"""
	Input:
		Data: [[InChI0,protein_sequence0,label0],[InChI1,protein_sequence1,label1],...]
	Output:
		return feature matrix X (N x 300), label list y, and a list l representing valid index of the Data;
		note that since compounds can be invalid, the returned (X,y) may be less than input Data;
		a txt file indicates failed data will be saved 
	"""
	dictionary = corpora.Dictionary.load('dict_for_1_nofeatureinvariant2.dict')
	tfidf = models.tfidfmodel.TfidfModel.load('tfidf_for_1_nofeatureinvariant2.tfidf')
	lsi = models.lsimodel.LsiModel.load('lsi_for_1_nofeatureinvariant2.lsi')
	model = Word2Vec.load('new_word2vec_model')

	X = []
	y = []
	l = []
	counter = 0
	f = open('failed_data.txt','wb')
	for i in Data:
		tmp = np.zeros(300)
		c = compute_compound_feature_(i[0], dictionary, tfidf, lsi)
		if c == 'error':
			f.writelines('line:'+str(counter+1)+'failed compound:'+str(Data[counter][0])+'\n')
			continue
		l.append(counter)
		p = compute_protein_feature_(i[1], model)
		tmp[:200] = c
		tmp[200:] = p
		X.append(tmp)
		y.append(i[2])
		counter += 1
	return X,y,l

def compute_feature_for_dataset_without_label(Data):
	"""
	Input:
		Data: [[InChI0,protein_sequence0],[InChI1,protein_sequence1],...]
	Output:
		return feature matrix X (N x 300) and a list representing valid index of the Data;
		note that since compounds can be invalid, the returned X may be less than input Data;
		a txt file indicates failed data will be saved
	"""
	dictionary = corpora.Dictionary.load('dict_for_1_nofeatureinvariant2.dict')
	tfidf = models.tfidfmodel.TfidfModel.load('tfidf_for_1_nofeatureinvariant2.tfidf')
	lsi = models.lsimodel.LsiModel.load('lsi_for_1_nofeatureinvariant2.lsi')
	model = Word2Vec.load('new_word2vec_model')

	X = []
	l = []
	counter = 0
	f = open('failed_data.txt','wb')
	for i in Data:
		print counter
		tmp = np.zeros(300)
		c = compute_compound_feature_(i[0], dictionary, tfidf, lsi)
		if c == 'error':
			f.writelines('line:'+str(counter+1)+'failed compound:'+str(Data[counter][0])+'\n')
			continue
		l.append(counter)
		p = compute_protein_feature_(i[1], model)
		tmp[:200] = c
		tmp[200:] = p
		X.append(tmp)
		counter += 1
	return X, l


def DeepCPI_train_and_predict(X_train, y_train, X_test):
	"""
	train on X_train (N,300), y_train (N,);
	predict on X_test (M, 300) 
	"""
	for ensemble in range(20):
		inputA = Input(shape=(200,))
		modelA = Dense(1024,W_regularizer=l1(0.0))(inputA)
		modelA = BatchNormalization()(modelA)
		modelA = Activation('relu')(modelA)
		modelA = Dropout(0.2)(modelA)
		modelA = Dense(256,W_regularizer=l1(0.0))(modelA)
		modelA = BatchNormalization()(modelA)
		modelA = Activation('relu')(modelA)
		modelA = Dropout(0.2)(modelA)
		inputB = Input(shape=(100,))
		modelB = Dense(1024,W_regularizer=l1(0.0))(inputB)
		modelB = BatchNormalization()(modelB)
		modelB = Activation('relu')(modelB)
		modelB = Dropout(0.2)(modelB)	
		modelB = Dense(256,W_regularizer=l1(0.0))(modelB)
		modelB = BatchNormalization()(modelB)
		modelB = Activation('relu')(modelB)
		modelB = Dropout(0.2)(modelB)	
		modelc = merge([modelA, modelB],mode='concat')
		modelc = Dense(512,W_regularizer=l1(0.0))(modelc)
		modelc = BatchNormalization()(modelc)
		modelc = Activation('relu')(modelc)
		modelc = Dropout(0.2)(modelc)	
		modelc = Dense(128,W_regularizer=l1(0.0))(modelc)
		modelc = BatchNormalization()(modelc)
		modelc = Activation('relu')(modelc)
		modelc = Dropout(0.2)(modelc)	
		modelc = Dense(32,W_regularizer=l1(0.0))(modelc)
		modelc = BatchNormalization()(modelc)
		modelc = Activation('relu')(modelc)
		modelc = Dropout(0.2)(modelc)	
		modelc = Dense(1)(modelc)
		modelc = Activation('sigmoid')(modelc)
		opt = Adagrad(0.01)
		model = Model(input=[inputA, inputB], output=modelc)
		model.compile(loss = 'binary_crossentropy',optimizer = opt,metrics=['accuracy'])
		print 'model compiled'

		#Suppose postive data is larger than the negative one, we downsample the positive data so that their number matched.
		positive_training_samples = []
		negative_training_samples = []
		for i in xrange(len(y_train)):
			if y_train[i] == 1:
				positive_training_samples.append(X_train[i])
			else:
				negative_training_samples.append(X_train[i])
		random_positive = np.random.randint(0,len(positive_training_samples),len(negative_training_samples))
		random_negative = np.random.randint(0,len(negative_training_samples),len(negative_training_samples))
		X_train_sampled = []
		y_train_sampled = []
		for i in xrange(len(random_positive)):
			X_train_sampled.append(positive_training_samples[random_positive[i]])
			y_train_sampled.append(1)
		for i in xrange(len(random_negative)):
			X_train_sampled.append(negative_training_samples[random_negative[i]])
			y_train_sampled.append(0)

		#training, epoch number should be tuned
		model.fit([np.array(X_train_sampled)[:,:200],np.array(X_train_sampled)[:,200:]],np.array(y_train_sampled),batch_size=512, nb_epoch=50)#,
		#predict
		pred = model.predict([np.array(X_test)[:,:200],np.array(X_test)[:,200:]],batch_size=512)

		if ensemble == 0:
			ensemble_pred = pred
		else:
			ensemble_pred += pred

	return ensemble_pred/float(20)




def DeepCPI_train_finetunemodel_and_predict(X_train, y_train, X_test):
	for ensemble in range(20):
		model = model_from_json(open('my_model_architecture'+str(ensemble)+'.json').read())
		model.load_weights('my_model_weights'+str(ensemble)+'.h5')
		model.compile(optimizer='adagrad', loss='binary_crossentropy')

		#Suppose postive data is larger than the negative one, we downsample the positive data so that their number matched.
		positive_training_samples = []
		negative_training_samples = []
		for i in xrange(len(y_train)):
			if y_train[i] == 1:
				positive_training_samples.append(X_train[i])
			else:
				negative_training_samples.append(X_train[i])
		random_positive = np.random.randint(0,len(positive_training_samples),len(negative_training_samples))
		random_negative = np.random.randint(0,len(negative_training_samples),len(negative_training_samples))
		X_train_sampled = []
		y_train_sampled = []
		for i in xrange(len(random_positive)):
			X_train_sampled.append(positive_training_samples[random_positive[i]])
			y_train_sampled.append(1)
		for i in xrange(len(random_negative)):
			X_train_sampled.append(negative_training_samples[random_negative[i]])
			y_train_sampled.append(0)

		#training, epoch number should be tuned
		model.fit([np.array(X_train_sampled)[:,:200],np.array(X_train_sampled)[:,200:]],np.array(y_train_sampled),batch_size=512, nb_epoch=20)#,
		#predict
		pred = model.predict([np.array(X_test)[:,:200],np.array(X_test)[:,200:]],batch_size=512)

		if ensemble == 0:
			ensemble_pred = pred
		else:
			ensemble_pred += pred
	return ensemble_pred/float(20)


def finetunemodel_predict(X_test):
	"""
	Load pretrained model and predict lables for X_test
	"""
	for ensemble in range(20):
		model = model_from_json(open('my_model_architecture'+str(ensemble)+'.json').read())
		model.load_weights('my_model_weights'+str(ensemble)+'.h5')
		model.compile(optimizer='adagrad', loss='binary_crossentropy')
		pred = model.predict([np.array(X_test)[:,:200],np.array(X_test)[:,200:]],batch_size=512)
		if ensemble == 0:
			ensemble_pred = pred
		else:
			ensemble_pred += pred
	return ensemble_pred/float(20)



def example_code_of_read_dict(filename):
	f = open(filename)
	tmp = pickle.load(f)
	f.close()
	for key,value in tmp.iteritems():
		print key,value

if __name__ == "__main__":
	example_data = './example_data.tsv'

	#load data
	Data = load_data_without_label(example_data)
	print ('data loaded')

	#compute features
	X, l = compute_feature_for_dataset_without_label(Data)
	print ('feature generated')

	#make predictions using pretrained model
	Y_pred = finetunemodel_predict(X)
	print ('prediction finished')

	#write predictions to file
	f = open('Prediction_results.tsv', 'wb')
	for i in range(len(l)):
		f.writelines(Data[l[i]][0]+'\t'+Data[l[i]][1]+'\t'+str(Y_pred[i][0])+'\n')
	f.close()
