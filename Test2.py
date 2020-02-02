from textgenrnn import textgenrnn

textgen = textgenrnn(weights_path='Test1_weights.hdf5',
                     vocab_path='Test1_vocab.json',
                     config_path='Test1_config.json')

textgen.generate_samples(max_gen_length=1000)
textgen.generate_to_file('textgenrnn_texts.txt', max_gen_length=1000)