
economics directory-> Contains the code leveraged to produce TF-IDF and Doc2Vec analysis
	clustering.py -> actual .py file that produces outputs stored in "results" directory.
	clustering_in.txt -> confermation of inputs in text format
	clustering_out.txt -> terminal output in text format
	
	results directory -> contains resulting graphics from analysis
	
	econ directory -> contains the aggregated and raw corpus files
		files directory -> combined random + econ corpora 
			all.jsonl -> master corpus (random + economcis)
		output_from_scrapy -> raw output from scrapy "items" files
			econ.jsonl -> items file from scrapy run
			random.jsonl -> items file from scrapy run
	