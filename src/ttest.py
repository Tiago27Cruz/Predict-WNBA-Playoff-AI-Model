import scipy.stats as stats

def read_result(filename):
    with open(filename) as file:
        return [float(line.rstrip()) for line in file]

def ttest(good_results, baseline_results, alpha):
    result = stats.ttest_ind(good_results, baseline_results, alternative="greater", equal_var=False)
    if (result.pvalue >= alpha):
        print("The alternative hypothesis is significant")
    else:
        print("The alternative hypothesis is not significant")

ttest(read_result("tests_gb.txt"), read_result("tests_baseline.txt"), 0.05)