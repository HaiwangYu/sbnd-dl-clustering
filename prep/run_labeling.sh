start_entry=$1
end_entry=$2

python labeling.py --tru-prefix tru-apa1 --rec-prefix rec-apa1 --out-prefix rec-lab-apa1 --entries ${start_entry}-${end_entry} | tee log
python labeling.py --tru-prefix tru-apa0 --rec-prefix rec-apa0 --out-prefix rec-lab-apa0 --entries ${start_entry}-${end_entry} | tee -a log