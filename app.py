import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Get absolute parent path
path = str(Path(__file__).parent.absolute())

# Extracting Kaggle Database source:https://www.kaggle.com/datasets/nikhil25803/github-dataset/data
githubDataFrame = pd.read_csv(path + "/github_dataset.csv")
repositoryDataFrame = pd.read_csv(path + "/repository_data.csv")

# View Raw Data
st.title('Beacon Technical Product Manager - Internship Challenge')
st.divider()

st.header('Kaggle GitHub Datasets')

# GitHub Dataset
st.subheader('GitHub Dataset')
st.markdown('''
    This raw dataset is a collection of 1052 GitHub repositories with at least 1 open issue. There are 1052 rows and 7 
    columns in the table. The table includes columns such as primary language used in the repository, fork count, open 
    pull requests, issue count, contributors etc.
''')
st.write("Shape: ", githubDataFrame.shape)
st.dataframe(githubDataFrame)

# Repository Dataset
st.subheader('Repository Dataset')
st.markdown('''
    This raw dataset is a collection of 2410866 GitHub repositories. There are 2917951 rows and 10 columns in the table.
    Along with certain columns present in the GitHub Dataset, the Repository Dataset also includes columns such as 
    licenses used, commit count, repository creation date etc." 
''')
st.write("Shape: ", repositoryDataFrame.shape)
st.dataframe(repositoryDataFrame[:25])
st.caption(':green[For visualization purposes, the above table is displaying only first 25 rows.]')
st.divider()

# Data Cleaning
st.header('Data Cleaning')
st.markdown('''
    Data cleaning is an essential part of data analysis. It is the process of fixing or removing incorrect, 
    corrupted, incomplete, or inconsistent data. This ensures the accuracy, reliability, consistency, and completeness 
    of the data.
''')
st.markdown("""
        **Importance of Data Cleaning:**
        - **Improves data quality:** Ensures you analyze accurate and reliable data, leading to trustworthy insights.
        - **Enhances analysis efficiency:** Clean data makes analysis faster and less prone to errors.
        - **Boosts model performance:** Machine learning models perform better with clean training data.
        """)
st.markdown('''
    For better readability all :red["null/none"] values in :blue["language"], :blue["primary_language"], and 
    :blue["languages_used"] columns were changed to :orange["No language specified"] values and duplicate rows were 
    removed in both datasets. 
''')
# Removing null values in primary language from both repositories
githubDataFrame.language.isnull().value_counts()
githubDataFrame.language = githubDataFrame.language.fillna('No language specified')
githubDataFrame = githubDataFrame.drop_duplicates()
repositoryDataFrame.primary_language.isnull().value_counts()
repositoryDataFrame.primary_language = repositoryDataFrame.primary_language.fillna('No language specified')
repositoryDataFrame.languages_used.isnull().value_counts()
repositoryDataFrame.languages_used = repositoryDataFrame.languages_used.fillna('No language specified')
repositoryDataFrame = repositoryDataFrame.drop_duplicates()
st.divider()

# View Column names
# st.write(githubDataFrame.columns)


# Data Preparation and Organization
st.header('Data Preparation')
st.markdown("""
    Data preparation transforms raw data into a format suitable for analysis.
    **Importance of Data Preparation:**
    - **Streamlines analysis:** Prepared data is easier to work with and analyze, saving time and effort.
    - **Enables consistent comparisons:** Standardized data allows for accurate comparisons across different datasets.
    - **Facilitates feature engineering:** Prepared data allows for extracting relevant features for analysis and model 
    building.
    """)
st.markdown('''
    In the GitHub dataset :blue["repositories"] column has been split into 2 separate columns - 
    :blue["repository_name"] and :blue["user_name"]. 
''')

githubDataFrame['repository_name'] = githubDataFrame.repositories.str.split('/').str[1]
githubDataFrame['user_name'] = githubDataFrame.repositories.str.split('/').str[0]
githubDataFrame = githubDataFrame.drop(['repositories'], axis=1)
githubDataFrame = githubDataFrame[githubDataFrame.columns.tolist()[-2:] + githubDataFrame.columns.tolist()[:-2]]

# Rename columns for readability
githubDataFrameOldColumns = githubDataFrame.columns
githubDataFrameNewColumns = ['Repository Name', 'User Name', 'Star Count', 'Fork Count', 'Issue Count', 'Pull Requests',
                             'Contributors', 'Language']
for index in range(len(githubDataFrameOldColumns)):
    githubDataFrame = githubDataFrame.rename(
        columns={githubDataFrameOldColumns[index]: githubDataFrameNewColumns[index]}
    )

repositoryDataFrameOldColumns = repositoryDataFrame.columns
repositoryDataFrameNewColumns = ['Name', 'Star Count', 'Fork Count', 'Watchers', 'Pull Requests', 'Primary Language',
                                 'Languages Used', 'Commit Count', 'Created At', 'License']
for index in range(len(repositoryDataFrameOldColumns)):
    repositoryDataFrame = repositoryDataFrame.rename(
        columns={repositoryDataFrameOldColumns[index]: repositoryDataFrameNewColumns[index]}
    )
st.subheader('Both Datasets after data cleaning and preparation look like this:')
st.dataframe(githubDataFrame)
st.write("Shape: ", githubDataFrame.shape)
st.caption(':green[GitHub Dataset]')
st.text('\n')
st.dataframe(repositoryDataFrame[:25])
st.write("Shape: ", repositoryDataFrame.shape)
st.caption(':green[Repository Dataset]')
st.divider()


# Data Analysis
st.header('Data Analysis')
st.markdown('''
    Now, extracting and interpreting meaningful insights from data using various analytical techniques:
''')

# Top 5 Repositories with most contributions
contributions = githubDataFrame.sort_values(by='Contributors', ascending=False)[:10]
st.subheader('Top 10 Repositories with most contributions')
st.markdown('''
    - :violet[**Most Popular Repository:**] The repository with the highest number of contributors is :orange["LinkFree"] with 658 
    contributors.
    - :violet[**GitHub - a Community Based Product:**] While GitHub is a wildly popular enterprise solution for version control,
    it can be visualized from the bar chart below that it is also a very popular :orange["Community Centric Product"]. 
    Many contributors add their improvements to these repositories and collaborate at a large scale.
    - :violet[**General Overview:**] The distribution of contributors across these top 10 repositories is quite varied, with some
     repositories having a significantly higher number of contributors than others. This could indicate the popularity, 
     activity, or community engagement level of these repositories.
''')
st.bar_chart(contributions, x="Repository Name", y="Contributors")

# Programming Language Usage Trend over the years on GitHub
st.subheader("Programming Language usage trend over the years on GitHub")
repositoryDataFrame['Year'] = repositoryDataFrame['Created At'].str.split('-').str[0]
lineChartDataFrame = repositoryDataFrame.groupby(['Year', 'Primary Language'], as_index=False)['Star Count'].count()
lineChartDataFrame = lineChartDataFrame.sort_values(['Year', 'Star Count'], ascending=[True, False]).groupby('Year').head(5)
lineChartDataFrame = pd.pivot_table(lineChartDataFrame, values='Star Count', index='Year', columns='Primary Language')
lineChartDataFrame = lineChartDataFrame.fillna(0)
lineChartDataFrame = lineChartDataFrame.reset_index()
# Omitting 2023 since does not have full year's data
lineChartDataFrame = lineChartDataFrame[lineChartDataFrame['Year'] != '2023']
st.markdown('''
    - :violet[**Python's Rise:**] :orange[Python's popularity] has seen a significant rise over the years, peaking around 2020. 
    This could be attributed to the surge in data science, machine learning, and AI projects, where Python is a 
    dominant language.
    - :violet[**JavaScript's Consistency:**] :orange[JavaScript] has remained consistently popular, reflecting its 
    central role in web development. It reached its peak between 2017 and 2019 but slightly declined after that.
    - :violet[**Jupyter Notebook's Introduction:**] The :orange[Jupyter Notebook] data indicates its emergence and 
    growth around 2014, aligning with the rise in data science and interactive computing trends.
    - :violet[**Decline of Traditional Languages:**] Languages like :orange[C], :orange[C++], and :orange[Java] show a 
    decline in recent years, suggesting a shift in the programming landscape. While they remain fundamental, newer 
    languages and technologies might be overshadowing them.
    - :violet[**Ambiguity with "No language specified”:**] The line representing "No language specified" suggests that 
    a significant number of repositories did not specify a language, especially around 2018-2020. 
    This could be due to various reasons such as documentation repositories, or repositories with non-code assets.

''')
st.line_chart(lineChartDataFrame, x="Year")
st.markdown('''
    The above graph has been contructed to show interesting insights of programming language usages over the years. For 
    the data to be constructed, the :blue["Created At] column in :blue["Repository Data"] has been split and the 
    creation :blue["Year"] of each repository is extracted. The rows are grouped by [Year, Primary Language] and sorted 
    based on [Year, Star Count].
''')
st.caption(':green[Here is the data for deeper visualization:]') ##
st.dataframe(lineChartDataFrame)

# Stars VS Forks Count
fig, ax = plt.subplots()
st.subheader('Stars VS Forks Count')
st.markdown('''
    Q: What its the difference between Starring a repository VS Forking a repository?
    Ans: :orange["Starring"] a repository is a way of bookmarking it for future reference, while :orange["Forking"] a 
    repository creates a copy that one can modify and contribute to.
''')
st.markdown('''
    - :violet[**High Concentration:**] There's a high concentration of repositories with a lower number of stars (0-200) and 
    forks (0-200). This suggests that many repositories receive a moderate amount of attention and engagement.
    - :violet[**Sparse Ist Quadrant (Upper Right):**] There are very few repositories with a high number of stars (>700) 
    and forks (>600), indicating that only a select few repositories achieve such high popularity.
    - :violet[**Few High-Star, Low-Fork Repos:**] Some repositories have a high number of stars but relatively fewer 
    forks (Stars: 995, Forks: 0). This suggests that there are repositories where users prefer to 'bookmark' the 
    repository but do not intent to make a copy and modify it.
    - :violet[**General Trend:**] In general, as the number of stars increases, the number of forks also tends to 
    increase, but not linearly. There's a broader spread in fork counts as star counts increase, indicating popular
    repositories have less modifications and thus lesser forking.
''')
st.scatter_chart(githubDataFrame, x="Star Count", y="Fork Count")


# Top 10 popular languages
st.subheader('Top 10 popular languages on GitHub')
languagesUsed = githubDataFrame.loc[githubDataFrame['Language'] != 'No language specified']
languagesUsed = languagesUsed['Language'].value_counts()[:10]
st.markdown('''
    - :violet[**Most Popular:**] :orange["JavaScript"] stands out as the most popular language on GitHub with a count of
    237, followed by Python at 151.
    - :violet[**Lower Popularity:**] Languages such as :orange["C++"], :orange["CSS"], :orange["Dart"], :orange["Ruby"],
    and :orange["Typescript"] have a relatively lower count, all in range 30 - 40, suggesting that while these are among
    the top 10, they are not as popular as the aforementioned languages.
    - :violet[**Web Development Dominance:**] Both :orange["JavaScript"] and :orange["HTML"] are key languages for 
    web development, and their high popularity suggests a significant amount of web development centric repositories on 
    GitHub.
    - :violet[**Data Science & Machine Learning:**] The presence of :orange["Python"] and :orange["Jupyter Notebook"] in
    the top languages highlights the growth and popularity of data science and machine learning projects on GitHub.
''')
st.caption(':green[For this graph repositories with no language specified have not been considered]')
st.bar_chart(languagesUsed)

st.markdown('The below graph is a broader representation of repositories across GitHub. When it comes to repositories '
            'without open issue requests, :orange["JavaScript"] and :orange["Python"] are equally popular languages '
            'for most repositories.')
st.bar_chart(repositoryDataFrame["Primary Language"].value_counts()[:10], color="#FFC300")

githubDataFrame = githubDataFrame.loc[githubDataFrame.Language != 'No language specified']
repositoryDataFrame = repositoryDataFrame.loc[repositoryDataFrame['Primary Language'] != 'No language specified']

# Languages with the Highest Star Count Repositories
st.subheader('Languages with the Highest Star Count Repositories')
sortedByStartCount = repositoryDataFrame.sort_values(by='Star Count', ascending=False)
languagesUsed = sortedByStartCount['Primary Language'].value_counts()[:10]
st.markdown('''
    - :violet[**Top Contenders:**] :orange["JavaScript"] and :orange["Python"] repositories have the highest star counts, approaching 
    450,000 stars. This suggests that they are among the most popular languages for the well-acknowledged repositories 
    on GitHub.
    - :violet[**Lower Popularity:**] Languages such as :orange["C"], :orange["C#"], :orange["C++"], :orange["Go"], and 
    :orange["Typescript"] have star counts below 150,000. While these repositories are still popular given their 
    inclusion in the top 10, they have relatively fewer stars compared to the leading languages.
    - :violet[**Web Development Recognition:**] The high star counts for :orange["JavaScript"], :orange["HTML"], and 
    :orange["PHP"] highlight the importance and recognition of web development repositories on GitHub.
    - :violet[**Emerging Languages:**] The presence of :orange["Go"] and :orange["Typescript"] in the list indicates the rising 
    popularity and adoption of these newer languages in the developer community.
''')
st.bar_chart(languagesUsed, color="#C70039")

# Languages with the Highest Fork Count Repositories
st.subheader('Languages with the Highest Fork Count Repositories')
sortedByForkCount = repositoryDataFrame.sort_values(by='Fork Count', ascending=False)
st.write(sortedByForkCount[:20])
languagesUsed = sortedByForkCount['Primary Language'].value_counts()[:10]
st.bar_chart(languagesUsed)

# Repositories with the Highest Issue Counts
st.subheader('Repositories with the Highest Issue Counts')
languagesUsed = githubDataFrame.sort_values(by='Issue Count', ascending=False)
languagesUsed = languagesUsed['Language'].value_counts()[:10]
st.bar_chart(languagesUsed)

# Repositories with the Highest Pull Requests
st.subheader('Repositories with the Highest Pull Requests')
languagesUsed = repositoryDataFrame.sort_values(by='Pull Requests', ascending=False)
languagesUsed = languagesUsed['Primary Language'].value_counts()[:10]
st.bar_chart(languagesUsed)

# Repositories with the Highest Commit Requests
st.subheader('Repositories with the Highest Commits ')
languagesUsed = repositoryDataFrame.sort_values(by='Pull Requests', ascending=False)
languagesUsed = languagesUsed['Primary Language'].value_counts()[:10]
st.bar_chart(languagesUsed)

# Forck, Issues, Commit, pull, watchers, license(top licesne, license vs valueCount)





# created, extrcat year, y axis group by count of primary language,

# Remove non for lang col in 2nd table , stars vs forks data set usage