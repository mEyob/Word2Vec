from gensim.models import word2vec
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

'''Thanks to https://rare-technologies.com/word2vec-tutorial/
for providing a memory efficient way of iterating sentences in a corpus
'''
class Corpus(object):
    def __init__(self, fname):
        self.fname = fname
 
    def __iter__(self):
        for line in open(self.fname):
            yield line.split()


#num_features =  Word vector dimensionality                      
#min_word_count = Minimum word count                        
#num_workers  = Number of threads to run in parallel
#context = Context window size                                                                                    
#downsampling = Downsample setting for frequent words

# sg = 0 (CBOW) sg = 1 Skip-gram
# hs = 0 Neg sampling hs = 1 hierarchical softmax

def trainModel(Arch, windowSize, downSampling, numFeatures):
	file = os.path.join(os.getcwd(),'Term-project/Data/wikipedia2008_en.txt')
	sentences = Corpus(file) # a memory-friendly iterator
	if Arch == 0:
		alg = 'CBOW'
	else:
		alg = 'SkipGram'
		
	print('Training model with parameters: ')
	print('\tArchitecture {}'.format(alg))
	print('\tWindow size {}'.format(windowSize))
	print('\tDown sampling {}'.format(downSampling))
	print('\tNumber of feature vectors {} \n'.format(numFeatures))
	
	model = word2vec.Word2Vec(sentences, sg=Arch, window = windowSize, \
           sample=downSampling, size=numFeatures, min_count = 20, workers=4)
			

	model.init_sims(replace=True)

	
	model_name = alg+'_ws'+ str(windowSize) + '_ds'+ str(downSampling) + '_ft' + str(numFeatures)
	model.save(model_name)
	model = None
	sentences = None