PERIOD_IN_MINS=10

while true;
do
    rsync -vzP -e ssh rasp-uk-tunnel:Coral-TPU/neural-networks/outputs/* ./outputs/
    sleep $((PERIOD_IN_MINS * 60))
done