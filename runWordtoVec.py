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


def test():
	fname = open("test.txt", "a")
	
	for i in range(100000):
		fname.write("test"+str(i)+"\n")
	fname.close()
		



#num_features = 300    # Word vector dimensionality                      
#min_word_count = 40   # Minimum word count                        
#num_workers = 4       # Number of threads to run in parallel
#context = 10          # Context window size                                                                                    
#downsampling = 1e-3   # Downsample setting for frequent words

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
			


	# If you don't plan to train the model any further, calling 
	# init_sims will make the model much more memory-efficient.
	model.init_sims(replace=True)

	# It can be helpful to create a meaningful model name and 
	# save the model for later use. You can load it later using Word2Vec.load()
	
	model_name = alg+'_ws'+ str(windowSize) + '_ds'+ str(downSampling) + '_ft' + str(numFeatures)
	model.save(model_name)
	model = None
	sentences = None