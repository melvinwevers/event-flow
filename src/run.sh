
start=`date +%s`
echo EVENT FLOW IS ALIVE

#python calculate_jump_entropy.py\
#    --input-file ../models/vk_100_model.pcl\

python pre_process.py\
    --newspaper parool\

python src/make_tm.py\
    --newspaper parool\
    --k 100

python calculate_jump_entropy.py\
    --input-file ../models/parool_100_model.pcl\

python pre_process.py\
    --newspaper nn\

python make_tm.py\
    --newspaper nn\
    --k 100

python calculate_jump_entropy.py\
    --input-file ../models/nn_100_model.pcl\

python pre_process.py\
    --newspaper nrc\

python make_tm.py\
    --newspaper nrc\
    --k 100

python calculate_jump_entropy.py\
    --input-file ../models/nrc_100_model.pcl\

kill $!; trap 'kill $!' SIGTERM
echo
echo ':)'

end=`date +%s`
runtime=$((end-start))
echo "Total time:" $runtime "sec"
