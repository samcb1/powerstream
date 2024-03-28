import streamlit as st
import numpy as np
import plotly.express as px
from scipy import stats

# Title
st.title('Sample Distribution Analysis')

# User inputs
POP_A = st.number_input('Population size for Group A', min_value=1, value=1200)
POP_B = st.number_input('Population size for Group B', min_value=1, value=800)
p_A_true = st.number_input('True probability for Group A', 0.0, 1.0, 0.85)
p_B_true = st.number_input('True probability for Group B', 0.0, 1.0, 0.90)
samples_A = st.number_input('Sample size for Group A', min_value=1, value=120)
samples_B = st.number_input('Sample size for Group B', min_value=1, value=80)
num_repeats = st.number_input('Number of bootstrap repeats', min_value=1, value=1000)
alpha = st.number_input('True probability for Group B', 0.0, 1.0, 0.01)

# Generating samples
pop_A = np.random.binomial(n=1, p=p_A_true, size=POP_A)
pop_B = np.random.binomial(n=1, p=p_B_true, size=POP_B)

def bootstrap_sample(population, samples_size, repeats):
    return [np.random.choice(population, size=(samples_size), replace=False).mean() for _ in range(repeats)]

def plot_histogram(data, true_mean=None,annot='', title='',xlabel=''):
    fig = px.histogram(data)
    if true_mean is not None:
        fig.add_vline(x=true_mean, line_dash="dash", annotation_text=annot, annotation_position="bottom right")
    if title != '':
        fig.update_layout(title=title)
    if xlabel != '':
        fig.update_xaxes(title=xlabel)
    st.plotly_chart(fig, use_container_width=True)

# Bootstrap sampling
sample_A_means = bootstrap_sample(pop_A, samples_A, num_repeats)
sample_B_means = bootstrap_sample(pop_B, samples_B, num_repeats)

# Plotting histograms
plot_histogram(sample_A_means, p_A_true,'Population mean', 'Sample A Means','Sample Yes %')
plot_histogram(sample_B_means, p_B_true,'Population mean', 'Sample B Means','Sample Yes %')

# Difference in Means
diff_means = np.array(sample_B_means)-np.array(sample_A_means)
plot_histogram(diff_means, title= 'Difference in Sample Means (B-A)',xlabel='B-A %')

# Proportion greater than 0
prop_greater_than_0 = (diff_means > 0).mean()
st.write(f'Proportion of difference in means greater than 0: {prop_greater_than_0}')

def z_test(p_A_sampled,N_A,p_B_sampled,N_B):
    P_pooled_sampled = ((p_A_sampled * N_A) + (p_B_sampled * N_B)) / (N_A + N_B)
    Z_sampled = (p_B_sampled - p_A_sampled) / ((P_pooled_sampled * (1 - P_pooled_sampled) * (1/N_A + 1/N_B)) ** 0.5)
    return 1 - stats.norm.cdf(Z_sampled)

p_values = [z_test(a,samples_A,b,samples_B) for a,b in zip(sample_A_means,sample_B_means)]
plot_histogram(p_values,alpha,'Alpha Value','Power demonstration',xlabel='p-values')
power = np.mean([p < alpha for p in p_values])
st.write(f'With an alpha of {alpha}, and the specified sample size this comparison of proportions has power of {np.round(power*100,2)}%')
st.write(f'Alpha represents the probability of a type I error. If the True probabilies were the same the test would say they were different 5% of the time')
st.write(f'Beta represents the probability of a type II error. If B were larger than A the test would reject this {np.round((1-power)*100,2)}% of the time')