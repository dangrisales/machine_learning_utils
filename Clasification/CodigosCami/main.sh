_mydir="$(pwd)"

PATH_features=${_mydir}"/features/"
PATH_results=${_mydir}"/results/"


for file_f in "lsaverbavg_motor_retell" "lsaverbavg_neutral_retell" 
    do

    python svm.py ${PATH_features}${file_f}".csv" ${PATH_results}${file_f}"_PDvsHC" 0
    python svm.py ${PATH_features}${file_f}".csv" ${PATH_results}${file_f}"_PDdclvsHCdcl" 1
    python svm.py ${PATH_features}${file_f}".csv" ${PATH_results}${file_f}"_PDnodclvsHCndcl" 2
    python svm.py ${PATH_features}${file_f}".csv" ${PATH_results}${file_f}"_PDdclvsPDnodcl" 3
done

