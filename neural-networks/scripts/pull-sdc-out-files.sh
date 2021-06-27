PERIOD_IN_MINS=5

cd ../outputs/

while true;
do
    SDC_OUT_FILES_BEFORE=$(ls -l | wc -l)
    rsync -vzP -e ssh rasp-uk-tunnel:Coral-TPU/neural-networks/outputs/* ./
    SDC_OUT_FILES_AFTER=$(ls -l | wc -l)
    NEW_SDC_OUT_FILES=$((SDC_OUT_FILES_AFTER - SDC_OUT_FILES_BEFORE))
    echo "Total SDC output files: $SDC_OUT_FILES_AFTER"
    echo "New SDC output files: $NEW_SDC_OUT_FILES"
    sleep $((PERIOD_IN_MINS * 60))
done
