data_dir=data

hw1:
	python dtree.py --train $(data_dir)/hw1/noisy10_train.ssv --test $(data_dir)/hw1/noisy10_test.ssv --valid $(data_dir)/hw1/noisy10_valid.ssv



hw2:
	python nbayes.py


hw3:
	python chain.py --n-var 4 --kind '01' --epsilon 0.01
	python chain.py --n-var 4 --kind '01' --epsilon 0.1
	python chain.py --n-var 4 --kind '01' --epsilon 0.2
	python chain.py --n-var 4 --kind '01' --epsilon 0.4
	python chain.py --n-var 10 --kind '01'
	python chain.py --n-var 100 --kind '01'
	python chain.py --n-var 1000 --kind '01'
	python chain.py --n-var 2 --kind 'rand'
	python chain.py --n-var 10 --kind 'rand'
	python chain.py --n-var 100 --kind 'rand'
	python chain.py --n-var 1000 --kind 'rand'
	python chain.py --n-var 10 --kind 'even'
	python chain.py --n-var 100 --kind 'even'
	python chain.py --n-var 1000 --kind 'even'

hw3_test:
	py.test tests/

demos_dominos:
	python -m demos.dominos


hw3_logreg: hw3_logreg_calc hw3_logreg_plot

hw3_logreg_calc:
	python logreg.py --calc

hw3_logreg_plot:
	python logreg.py --plot
