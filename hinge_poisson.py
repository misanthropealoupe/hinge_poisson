import json
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime

from scipy.stats import poisson
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('matches_json_path', help='Path to matches.json')
parser.add_argument('nbins', type=int, help='Number of temporal bins in which to place like/match activity. Hint: try setting nbins to the number of active months')
parser.add_argument('--confidence', type=float, default=0.9, help='Confidence interval of error bars')

args = parser.parse_args()
datafile = args.matches_json_path
nbins = args.nbins
match_conf_interval = args.confidence

# TODO: replace all poisson statistics with the appropriate binomials! Used for convenience

with open(datafile, 'r') as f:
    matches_raw = json.loads(f.read())

def parse_time(hingetime):
#     date, hms = hingetime.split('T')
    t = datetime.datetime.strptime(hingetime.split('.')[0], '%Y-%m-%dT%H:%M:%S')
    return t.timestamp()
    
class Match:
    def __init__(self, hingematch):
        self.rawmatch = hingematch.copy()
        
        keys = self.rawmatch.keys()
        self.match = 'match' in keys
        self.like = 'like' in keys
        self.has_chat = 'chat' in keys
        self.block = 'block' in keys
        
        assert self.block or self.match or self.like
        
        self.sorted_events = []
        for event in self.rawmatch:
            ar = self.rawmatch[event]
            tmin = parse_time(ar[0]['timestamp'])
            tmax = tmin
            for dic in ar:
                this_t = parse_time(dic['timestamp'])
                tmin = min(tmin, this_t)
                tmax = max(tmax, this_t)
            self.sorted_events.append([tmin, event])
        
        self.sorted_events = sorted(self.sorted_events, key=lambda x: x[0])
        
        self.t_events = {}
        for t, event in self.sorted_events:
            self.t_events[event] = t

        self.initiated = None
        if self.match:
            self.initiated = 'match' == self.sorted_events[0][1]
        
        self.has_comment = None
        if self.like:
            self.has_comment = 'comment' in self.rawmatch['like'][0]
        
        self.t_like_match = None
        if self.match and self.like:
            self.t_like_match = self.t_events['match'] - self.t_events['like']
        
        self.t_match = None
        if self.match:
            self.t_match = self.t_events['match']
        
        self.t_like = None
        if self.like:
            self.t_like = self.t_events['like']
        
        self.t_block = None
        if self.block:
            self.t_block = self.t_events['block']
            
        self.t0 = None # a representative time for this event
        
        # these exhaust all cases; self.t0 is guaranteed to
        # not be None due to an earlier assert
        if self.like:
            self.t0 = self.t_like
        elif self.match:
            self.t0 = self.t_match
        elif self.block:
            self.t0 = self.t_block        
        
matches = []
for m in matches_raw:
    match = Match(m)
    matches.append(match)

def count_prop(matches, kw):
    s = 0
    for m in matches:
        s += m.__dict__[kw]
    return s

def select_where(matches, kw, val):
    ret = []
    for m in matches:
        if m.__dict__[kw] == val:
            ret.append(m)
    return ret

matched = select_where(matches, 'match', True)
liked = select_where(matches, 'like', True)
matchlike = select_where(matched, 'like', True)
likecomment = select_where(liked, 'has_comment', True)
likenocomment = select_where(liked, 'has_comment', False)
matchcomment = select_where(matchlike, 'has_comment', True)
matchnocomment = select_where(matchlike, 'has_comment', False)
n_matchlike = len(matchlike)
n_matchcomment = len(matchcomment)
n_matchnocomment = len(matchnocomment)
n_likecomment = len(likecomment)
n_likenocomment = len(likenocomment)
n_like = len(liked)
n_initiated = count_prop(matched, 'initiated')
t_like_match_ave = count_prop(matchlike, 't_like_match')/n_matchlike
     
p_match = n_matchlike/n_like
p_matchcomment = n_matchcomment/n_likecomment
p_matchnocomment = n_matchnocomment/n_likenocomment
    
n_tot = len(matches)
n_match = count_prop(matches, 'match')
print(f'n_tot: {n_tot}, n_match: {n_match}')
print(f'p_match: {p_match:.3f}')
print(f'p_like_comment, {n_likecomment/n_like:.3f}')
print(f'p_match_comment: {p_matchcomment:.3f}')
print(f'p_match_no_comment: {p_matchnocomment:.3f}')

# do a simple variance analysis to determine whether like comments help/hinder you
# can't consider "comment bias" e.g. the tendency to leave comments for more
# attractive matches where a mutual match may be less likely

trial_factor = 4 # ignored, for now (should be considered when forming a p threshold)
# thresh = 0.05 / trial_factor

if p_matchnocomment < p_match:
    pval_nocomment = 1 - poisson.cdf(n_matchnocomment, p_match * n_likenocomment)
    nocomment_hypothesis = 'fewer'
else:
    pval_nocomment = poisson.cdf(n_matchnocomment, p_match * n_likenocomment)
    nocomment_hypothesis = 'more'
    
print(f'probability that the no comment like strategey results in {nocomment_hypothesis} matches: {pval_nocomment:.3f}')

if p_matchcomment < p_match:
    # here the cdf actually represents the probability the result is explained
    # by the null hypothesis (no difference)
    pval_comment = 1 - poisson.cdf(n_matchcomment, p_match * n_likecomment)
    comment_hypothesis = 'fewer'
else:
    pval_nocomment = poisson.cdf(n_matchcomment, p_match * n_likecomment)
    comment_hypothesis = 'more'
    
print(f'probability that the comment like strategey results in {comment_hypothesis} matches: {pval_nocomment:.3f}')


# print(f'p_match_like_comment: {}')
# print(f'p_match_like_nocomment: {}')
print(f'p_counterparty_initiated: {n_initiated/n_match:.3f}')
print(f'average like-to-match time (day): {t_like_match_ave/3600./24:.3f}')

# study the distribution of like-to-match time

t_lm = []
for m in matchlike:
    this_t = m.t_like_match / 3600.
    t_lm.append(this_t)
    
t_lm = np.array(t_lm)

cdf_x = np.linspace(0,1,128)
t_lm_cdf = np.quantile(t_lm, cdf_x)

t_lm_med = np.quantile(t_lm, 0.5)

t_lm_hist = t_lm[t_lm < 3*24.]

hist, edges = np.histogram(t_lm_hist, bins = 100)
plt.figure(dpi=300, facecolor='white')
plt.title('Distribution of Like-to-Match Time')
plt.xscale('log')
plt.bar(edges[:-1], hist, width=1. * (edges[1] - edges[0]))
plt.axvline(t_lm_med, label='median', color='red')
plt.legend()
plt.ylabel('matches')
plt.xlabel('like-to-match time (hours)')
plt.savefig('plots/like_match_dist.png')
plt.close()

plt.figure(dpi=300, facecolor='white')
plt.title('Like-to-Match Time CDF')
plt.xscale('log')
plt.plot(t_lm_cdf, cdf_x)
plt.axvline(t_lm_med, label='median', color='red')
# plt.axhline(0.5, color='red')
plt.legend()
plt.ylabel('propotion of all matches')
plt.xlabel('like-to-match time (hours)')
plt.grid(which='both')
plt.savefig('plots/like_match_cdf.png')
plt.close()

def get_t_range(matches, kw):
    t_min = matches[0].__dict__[kw]
    t_max = matches[0].__dict__[kw]
    
    for m in matches:
        t_min = min(t_min, m.__dict__[kw])
        t_max = max(t_max, m.__dict__[kw])
    
    return [t_min, t_max]


def bin_linear(matches, kw, nbins=10, t_range=None, edges=None):
    matches = sorted(matches, key=lambda x: x.__dict__[kw])
    
    assert nbins is not None or edges is not None
    
    ts = []
    for m in matches:
        ts.append(m.__dict__[kw])
    
    t_min = ts[0]
    t_max = ts[-1]

    if nbins is not None:
        if t_range is not None:
            assert t_range[1] >= t_range[0]
            dt = t_range[1] - t_range[0]
        else:
            dt = t_max - t_min
        edges = np.arange(nbins + 1) * dt / nbins  + t_min
        
    assert t_min >= edges[0] and t_max <= edges[-1]
    
    binned = []
    nmatch = len(matches)
    imatch = 0
    for i in range(nbins):
        tl = edges[i]
        tr = edges[i + 1]

        events = []
        if imatch < nmatch:
            while matches[imatch].__dict__[kw] < tr:
                events.append(matches[imatch])
                imatch += 1
                if imatch == nmatch:
                    break
            
        binned.append(events)

    return edges, binned

def count_bins(bins):
    nbins = len(bins)
    ret = np.zeros(nbins)
    
    for i in range(nbins):
        ret[i] = len(bins[i])
    
    return ret

# analyze like rate in binned time periods of fixed width
# consider cross-analyzing with number of likes as well
# dt_like = 3600. * 24 * 7 # one week window scale

ml_range = get_t_range(matchlike, 't_like')
l_range = get_t_range(liked, 't_like')

t_range = [min(ml_range[0], l_range[0]), max(ml_range[1], l_range[1])]

edges, binned_matchlike = bin_linear(matchlike, 't_like', nbins=nbins, t_range=t_range)
edges2, binned_match = bin_linear(matchlike, 't_match', nbins=nbins, t_range=t_range)
edges, binned_like = bin_linear(liked, 't_like', nbins=nbins, t_range=t_range)
x = 0.5 * (edges[:-1] + edges[1:])

n_ml_bin = count_bins(binned_matchlike)
n_l_bin = count_bins(binned_like)
n_match_bin = count_bins(binned_match)

binned_rates = n_ml_bin / n_l_bin
bin_width = x[1] - x[0]
bin_width_day = bin_width / 24. / 3600.
bin_width2 = edges2[1] - edges2[0]
bin_width2_day = bin_width2 / 24. / 3600.

# import matplotlib.dates as md

plt.figure(dpi=300, facecolor='white')
plt.title('Like Activity')
plt.bar(edges[:-1], n_l_bin/bin_width_day, width=x[1] - x[0])
plt.ylabel('number of likes sent per day')
plt.xlabel('time')
xticks = plt.xticks()[0]
xlabels = [datetime.datetime.fromtimestamp(t).strftime('%m-%y') for t in xticks]
ax = plt.gca()
ax.set_xticklabels(xlabels)
plt.savefig('plots/like_activity.png')

plt.figure(dpi=300, facecolor='white')
plt.title('Match Activity')
plt.bar(edges2[:-1], n_match_bin/bin_width2_day, width=bin_width2)
plt.ylabel('number of matches per day')
plt.xlabel('time')
xticks = plt.xticks()[0]
xlabels = [datetime.datetime.fromtimestamp(t).strftime('%m-%y') for t in xticks]
ax = plt.gca()
ax.set_xticklabels(xlabels)
plt.savefig('plots/match_activity.png')

yerr = np.empty((2,nbins))
for i in range(nbins):
    rate = binned_rates[i]
    
    n_ml = n_ml_bin[i]
    n_l = n_l_bin[i]

    interval = poisson.interval(match_conf_interval, n_ml) / n_l
    yerr[:,i] = [rate - interval[1], interval[0] - rate]

plt.figure(dpi=300, facecolor='white')
plt.title('Per-Bin Match Probabilities and 90% Error Intervals')
plt.errorbar(edges[:-1], binned_rates, yerr=yerr)
plt.axhline(p_match, color='red', label='overall match rate')
xticks = plt.xticks()[0]
xlabels = [datetime.datetime.fromtimestamp(t).strftime('%m-%y') for t in xticks]
ax = plt.gca()
ax.set_xticklabels(xlabels)
plt.ylabel('Match Probability in Bin')
plt.legend()
plt.savefig('plots/local_match_probability.png')