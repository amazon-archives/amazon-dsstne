BEGIN {
    FS = ","
    RS = "\r|\n|\r\n|\n\r"
    ORS = "\n"
}

NR > 1 {
    if (!prev) {
        printf $1 "\t"
    } else if (prev != $1) {
        printf "\n" $1 "\t"
    } else { 
        printf ":"
    }
    printf $2 "," $4
    prev = $1
}

END {
    printf "\n"
}
