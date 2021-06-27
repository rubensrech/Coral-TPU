#!/bin/bash

INTERVAL_IN_MINS=1

send_email() {
	SUBJ=$1
	CONT=$2
	osascript /Users/rubensrechjunior/Library/Mobile\ Documents/com\~apple\~ScriptEditor2/Documents/sendEmail.scpt "$SUBJ" "$CONT"
}

BEAM_STATUS=""

while true;
do
	URL="http://shadow.nd.rl.ac.uk/ChipIrWebServices/NeutronCounts"

	curl -s $URL | grep "OPEN" > /dev/null
	RET_VAL=$?

	PREV_STATUS=$BEAM_STATUS
	BEAM_STATUS=$(test $RET_VAL -eq 0 && echo "OPEN" || echo "CLOSED")

	if [ "$BEAM_STATUS" != "$PREV_STATUS" ]; then
		echo "STATUS CHANGED: $BEAM_STATUS"
		date
		send_email "Beam status changed: $BEAM_STATUS" "Current beam status: $BEAM_STATUS"
	fi
	
	sleep $((INTERVAL_IN_MINS * 60))
done
