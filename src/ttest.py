import scipy.stats as stats

def ttest(good_results, baseline_results, alpha):
    result = stats.ttest_ind(good_results, baseline_results, alternative="greater", equal_var=False)
    if (result.pvalue >= alpha):
        print("The alternative hypothesis is significant")
    else:
        print("The alternative hypothesis is not significant")