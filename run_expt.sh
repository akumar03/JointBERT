

run_jb_main  () {
task=$1
noise=$2
rm summary_${task}_rand_${noise}.tsv
for seed in 1 2 3
do
  rm -Rf data/${task}_rand_${noise}/
  rm data/cached_*
  python generate_random_noise.py --task $task --noise $noise --seed $seed
  python3 main.py --task ${task}_rand_${noise} --model_type bert  --model_dir ${task}_rand_${noise}_model  --do_train --do_eval --num_train_epochs 100 --learning_rate 5e-5
done
}

for noise in 10 20 30 40 50
do
  run_jb_main $1 $noise
done
