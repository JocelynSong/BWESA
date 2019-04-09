emb_file="/home/develop/Documents/AILab/Song/bwe/exp_data/AER/bible_eu_embs"
eval_data="/home/develop/Documents/AILab/Song/bwe/exp_data/AER/eval_data"

result=`python3 alignment_eval.py $emb_file/en-fr/en.emb $emb_file/en-fr/fr.emb $eval_data/graca/enfr/alignment $eval_data/graca/enfr/test.en $eval_data/graca/enfr/test.fr | tr -d '[[:space:]]'`
echo -e "graca\tenfr\ten\tfr\t$result"
result=`python3 alignment_eval.py $emb_file/en-fr/en.emb $emb_file/en-fr/fr.emb $eval_data/graca/enfr/alignment $eval_data/graca/enfr/test.en $eval_data/graca/enfr/test.fr R | tr -d '[[:space:]]'`
echo -e "graca\tenfr\tfr\ten\t$result"
result=`python3 alignment_eval.py $emb_file/en-es/en.emb $emb_file/en-es/es.emb $eval_data/graca/enes/alignment $eval_data/graca/enes/test.en $eval_data/graca/enes/test.es | tr -d '[[:space:]]'`
echo -e "graca\tenes\ten\tes\t$result"
result=`python3 alignment_eval.py $emb_file/en-es/en.emb $emb_file/en-es/es.emb $eval_data/graca/enes/alignment $eval_data/graca/enes/test.en $eval_data/graca/enes/test.es R | tr -d '[[:space:]]'`
echo -e "graca\tenes\tes\ten\t$result"
result=`python3 alignment_eval.py $emb_file/en-pt/en.emb $emb_file/en-pt/pt.emb $eval_data/graca/enpt/alignment $eval_data/graca/enpt/test.en $eval_data/graca/enpt/test.pt | tr -d '[[:space:]]'`
echo -e "graca\tenpt\ten\tpt\t$result"
result=`python3 alignment_eval.py $emb_file/en-pt/en.emb $emb_file/en-pt/pt.emb $eval_data/graca/enpt/alignment $eval_data/graca/enpt/test.en $eval_data/graca/enpt/test.pt R | tr -d '[[:space:]]'`
echo -e "graca\tenpt\tpt\ten\t$result"
result=`python3 alignment_eval.py $emb_file/en-fr/en.emb $emb_file/en-fr/fr.emb $eval_data/hansards/alignment $eval_data/hansards/test.en $eval_data/hansards/test.fr | tr -d '[[:space:]]'`
echo -e "hansards\ten\tfr\t$result"
result=`python3 alignment_eval.py $emb_file/en-fr/en.emb $emb_file/en-fr/fr.emb $eval_data/hansards/alignment $eval_data/hansards/test.en $eval_data/hansards/test.fr R | tr -d '[[:space:]]'`
echo -e "hansards\tfr\ten\t$result"
result=`python3 alignment_eval.py $emb_file/en-es/en.emb $emb_file/en-es/es.emb $eval_data/lambert/alignment $eval_data/lambert/test.en $eval_data/lambert/test.es | tr -d '[[:space:]]'`
echo -e "lambert\ten\tes\t$result"
result=`python3 alignment_eval.py $emb_file/en-es/en.emb $emb_file/en-es/es.emb $eval_data/lambert/alignment $eval_data/lambert/test.en $eval_data/lambert/test.es R | tr -d '[[:space:]]'`
echo -e "lambert\tes\ten\t$result"
result=`python3 alignment_eval.py $emb_file/en-ro/en.emb $emb_file/en-ro/ro.emb $eval_data/mihalcea/alignment $eval_data/mihalcea/test.ro $eval_data/mihalcea/test.en | tr -d '[[:space:]]'`
echo -e "mihalcea\tro\ten\t$result"
result=`python3 alignment_eval.py $emb_file/en-ro/en.emb $emb_file/en-ro/ro.emb $eval_data/mihalcea/alignment $eval_data/mihalcea/test.ro $eval_data/mihalcea/test.en R | tr -d '[[:space:]]'`
echo -e "mihalcea\ten\tro\t$result"
result=`python3 alignment_eval.py $emb_file/en-sv/en.emb $emb_file/en-sv/sv.emb $eval_data/holmqvist/alignment $eval_data/holmqvist/test.en $eval_data/holmqvist/test.sv | tr -d '[[:space:]]'`
echo -e "holmqvist\ten\tsv\t$result"
result=`python3 alignment_eval.py $emb_file/en-sv/en.emb $emb_file/en-sv/sv.emb $eval_data/holmqvist/alignment $eval_data/holmqvist/test.en $eval_data/holmqvist/test.sv R | tr -d '[[:space:]]'`
echo -e "holmqvist\tsv\ten\t$result"