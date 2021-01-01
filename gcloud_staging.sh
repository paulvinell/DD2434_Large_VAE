export TIMESTAMP=`date "+%Y%m%d%H%M%S"`
export STORAGE_NAME="project_large_vae_dd2434"
export JOB_NAME="large_vae_experiment_$TIMESTAMP"
export APP_PACKAGE_PATH=./large_vae
export MAIN_APP_MODULE=main # name of the training file
export JOB_DIR="gs://$STORAGE_NAME/jobs/$JOB_NAME"

# The below line submits the job to google cloud.
# The added flags will be passed as arguments to our code (and can be read with argparse)
# The example flags demonstrate how you can add your own flags.
gcloud ai-platform jobs submit training $JOB_NAME \
        --package-path $APP_PACKAGE_PATH \
        --module-name $MAIN_APP_MODULE \
        --job-dir $JOB_DIR \
        --region europe-west1 \
        --config gcloud/config.yaml \
        -- example-flag-1 666 \
        -- example-flag-2 "value"
