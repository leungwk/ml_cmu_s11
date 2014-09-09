data_dir=data

hw1:
	python dtree.py --train $(data_dir)/hw1/noisy10_train.ssv --test $(data_dir)/hw1/noisy10_test.ssv --valid $(data_dir)/hw1/noisy10_valid.ssv
