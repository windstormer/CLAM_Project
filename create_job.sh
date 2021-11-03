#! /bin/bash

# set an initial value for the flag
job_name=tmp_job
account=MST110161
nodes=1
gpu=1
cpt=4
cpg=4
ntasks_per_node=1
time_limit=96:00:00
mail=
mail_type=ALL

partition=gp4d  ## gtest (testing), gp1d (at most one day), gp2d (at most two days), gp4d (at most four days)
nodelist=

# read the options
TEMP=`getopt -o j:a:m:n:g:c:t:p: --long job-name:,account:,partition:,ntasks-per-node:,nodelist:,mail:,mail-type:,nodes:,gpu:,cpus-per-gpu:,cpus-per-task:,time-limit:,path: -n 'test.sh' -- "$@"`
eval set -- "$TEMP"

# extract options and their arguments into variables.
while true ; do
    case "$1" in
        -j|--job-name) job_name=$2 ; shift 2 ;;
        -a|--account) account=$2 ; shift 2 ;;
        -m|--mail) mail=$2 ; shift 2 ;;
        --mail-type) mail_type=$2 ; shift 2 ;;
        -n|--nodes) nodes=$2 ; shift 2 ;;
        -g|--gpu) gpu=$2 ; shift 2 ;;
        -c) cpt=$2 ; cpg=$2 ;  shift 2 ;;
		--nodelist) nodelist=$2 ; shift 2 ;;
		--partition) partition=$2 ; shift 2 ;;
		--ntasks-per-node) ntasks_per_node=$2 ; shift 2 ;;
        --cpus-per-gpu) cpg=$2 ; shift 2 ;;
        --cpus-per-task) cpt=$2 ; shift 2 ;;
        -t|--time-limit) time_limit=$2 ; shift 2 ;;
        -p|--path) path=$2 ; shift 2 ;;
        --) shift ; break ;;
        *) echo "Internal error!" ; exit 1 ;;
    esac
done

output=../job_logs/${job_name}_$(date +"%m_%d_%H_%M").out

echo "job_name = $job_name"
echo "account = $account"
echo "mail = $mail"
echo "mail_type = $mail_type"
echo "nodes = $nodes"
echo "partition = $partition"
echo "ntasks_per_node = $ntasks_per_node"
echo "gpu = $gpu"
echo "cpt = $cpt"
echo "cpg = $cpg"
echo "time_limit = $time_limit"
echo "output = $output"
if [ -z "$nodelist" ]
then
      echo "Nodelist Not Specified."
else
      echo "nodelist = $nodelist"
	  nodelist=--nodelist=$nodelist
fi

if [ -z ${path} ]
then
    printf -- '-%.0s' {1..20}; echo ""
    echo "Run script does not be specified yet."
    echo "Use -p [script_path] or --path [script_path] to provide the script path."
    printf -- '-%.0s' {1..20}; echo ""
    exit 1 
else
    echo "path = $path"
fi

sbatch --job-name=$job_name --account=$account --mail-user=$mail --mail-type=$mail_type --partition=$partition --nodes=$nodes $nodelist --gres=gpu:$gpu --cpus-per-task=$cpt --cpus-per-gpu=$cpg --ntasks-per-node=$ntasks_per_node --time=$time_limit --output=$output srun_job.sh $path
