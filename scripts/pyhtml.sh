#!/usr/bin/bash
for i in ../*.py; do vim -c TOhtml -c wqa ./$(basename ${i}) ; done
