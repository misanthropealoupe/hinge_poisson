# Installation:

pip install -r requirements.txt

# Usage:

python3 hinge_poisson.py \<-h\> \<matches_json_path\> \<nbins\> \<--confidence=0.9\>

(run from the repo directory)

## Arguments:

**matches_json_path:** absolute or relative path to your matches.json file

**nbins:** the number of temporal bins in which to group like/match activity. Play around with different values; I find that roughly nbins = n_months_active is a good choice. Setting nbins too high will result in a very large error bars on match rate, whereas setting it too low will result in poor ability to spot trends in like/match rate over time

**confidence:** the confidence interval of the error bars on temporal match rate

## Outputs:

Output plots reside in \<repo_dir\>/plots

