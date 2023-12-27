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
st.write("Rows x Columns:: ", githubDataFrame.shape)
st.dataframe(githubDataFrame)

# Repository Dataset
st.subheader('Repository Dataset')
st.markdown('''
    This raw dataset is a collection of 2410866 GitHub repositories. There are 2917951 rows and 10 columns in the table.
    Along with certain columns present in the GitHub Dataset, the Repository Dataset also includes columns such as 
    licenses used, commit count, repository creation date etc." 
''')
st.write("Rows x Columns:: ", repositoryDataFrame.shape)
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
st.caption(':green[GitHub Dataset]')
st.write("Rows x Columns: ", githubDataFrame.shape)
st.dataframe(githubDataFrame)
st.text('\n')
st.caption(':green[Repository Dataset]')
st.write("Rows x Columns: ", repositoryDataFrame.shape)
st.dataframe(repositoryDataFrame[:25])
st.divider()


# Data Analysis
st.header('Data Analysis')
st.markdown('''
    Now, extracting and interpreting meaningful insights from data using various analytical techniques:
''')

# Top 10 Repositories with most contributions
contributions = githubDataFrame.sort_values(by='Contributors', ascending=False)[:10]
st.subheader('Top 10 Repositories with Most Contributions')
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
    - :violet[**Decline of PHP:**] :orange[PHP] once a dominant language for server-side web development, has seen a 
    decline post 2015, possibly due to the emergence of other backend technologies and frameworks.
    - :violet[**Ambiguity with "No language specified”:**] The line representing "No language specified" suggests that 
    a significant number of repositories did not specify a language, especially around 2018-2020. 
    This could be due to various reasons such as documentation repositories, or repositories with non-code assets.

''')
st.line_chart(lineChartDataFrame, x="Year")
st.markdown('''
    The above graph has been constructed to show interesting insights of programming language usages over the years. For 
    the data to be constructed, the :blue["Created At] column in :blue["Repository Dataset"] has been split and the 
    creation :blue["Year"] of each repository is extracted. The rows are grouped by [Year, Primary Language] and sorted 
    based on [Year, Star Count].
''')
st.caption(':green[Here is the data for deeper visualization:]')
st.dataframe(lineChartDataFrame)

# Stars VS Forks Count
fig, ax = plt.subplots()
st.subheader('Stars VS Forks Count')
st.markdown('''
    :pink[**Ques:**] What its the difference between Starring a repository VS Forking a repository?
    
    :pink[**Ans:**] :orange["Starring"] a repository is a way of bookmarking it for future reference, while 
    :orange["Forking"] a repository creates a copy that one can modify and contribute to.
''')
st.markdown('''
    - :violet[**High Concentration:**] There's a high concentration of repositories with a lower number of stars (0-200) 
    and forks (0-200). This suggests that many repositories receive a moderate amount of attention and engagement.
    - :violet[**Sparse Ist Quadrant (Upper Right):**] There are very few repositories with a high number of stars (>700) 
    and forks (>600), indicating that only a select few repositories achieve such high popularity.
    - :violet[**High-Star, Low-Fork Repos:**] Some repositories have a high number of stars but relatively fewer 
    forks (Stars: 995, Forks: 0). This suggests that there are repositories where users prefer to 'bookmark' the 
    repository but do not intend to make a copy and modify it.
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
st.caption(':green[For this graph repositories with no language specified have not been considered and this graph '
           'represents repositories where there is at least 1 open issue count.]')
st.bar_chart(languagesUsed)

st.markdown('''
    The below graph is a broader representation of repositories across GitHub. Looking at all the repositories in the 
    :orange["Repository Dataset"] here are some key observations:
''')
st.markdown('''
    - :violet[**Top Contenders:**] :orange["JavaScript"] and :orange["Python"] repositories have the highest star counts
    , approaching 450,000 stars. This suggests that there are many repositories where :orange["Python/JavaScript"] 
    are the primary language with 0 open issue counts.
    - :violet[**Lower Popularity:**] Languages such as :orange["C"], :orange["C#"], :orange["C++"], :orange["Go"], and 
    :orange["Typescript"] are the primary language for <150,000 repositories.
    - :violet[**Web Development Recognition:**] :orange["JavaScript"], :orange["HTML"], and :orange["PHP"] highlight the 
    importance and recognition of web development repositories on GitHub.
    - :violet[**Emerging Languages:**] The presence of :orange["Go"] and :orange["Typescript"] in the list indicates the 
    rising popularity and adoption of these newer languages in the developer community.
''')
languagesUsed = repositoryDataFrame.loc[repositoryDataFrame['Primary Language'] != 'No language specified']
languagesUsed = languagesUsed["Primary Language"].value_counts()[:10]
st.bar_chart(languagesUsed, color="#FFC300")

githubDataFrame = githubDataFrame.loc[githubDataFrame.Language != 'No language specified']
repositoryDataFrame = repositoryDataFrame.loc[repositoryDataFrame['Primary Language'] != 'No language specified']

# Top 10 popular Licenses used in GitHub Repositories
st.subheader('Top 10 popular Licenses used in GitHub Repositories ')
st.markdown('''
    - :violet[**Most Popular License:**] It clear from the data that :orange["MIT License"] is the most popular license 
    used in GitHub repositories with >73,000 usage, followed by :orange["Apache License 2.0"].
''')
licenseUsed = repositoryDataFrame['License'].value_counts()[:10]
st.bar_chart(licenseUsed)

st.divider()

st.markdown('''
    Note: There are 6 additional graphs with their analyis. However, the streamlit server could not handle the page load
    and hence the same have been commented in the code. The screenshots of all the graphs including the additional 6 
    graphs are present in this GoogleDrive folder with view permissions.
    
    Drive link: https://drive.google.com/drive/folders/1DS7bnk_lEX8DAj58KSWyc7Vk44FZBdpY?usp=sharing
    GitHub: https://github.com/snigi-gupta/BeaconTechnicalPMInternshipChallenge/tree/main
''')
"""
# Repositories with the Highest Star Counts
starCount = repositoryDataFrame.sort_values(by='Star Count', ascending=False)[:10]
st.subheader('Repositories with Highest Star Counts')
st.markdown('''
    - :violet[**Web Development Dominance:**] Both :orange["Bootstrap"] and :orange["React"] are prominent web
    development tools, with "Bootstrap" being a popular frontend framework and "React" being a widely-used JavaScript
    library. Their high star counts suggest their significant impact and adoption in the web development community.
    - :violet[**API Resources:**] The :orange["public-apis"] repository, which likely provides a collection of free APIs
    for development and testing, is also popular, indicating a demand for such resources.
    - :violet[**System Design:**] :orange["system-design-primer"] is a repository that likely provides insights into
    system design, and its high star count suggests that system design is a topic of interest for many developers.
''')
st.bar_chart(starCount, x="Name", y="Star Count", color="#f67410")

# Repositories with the Highest Fork Count
forkCount = repositoryDataFrame.sort_values(by='Fork Count', ascending=False)[:10]
st.subheader('Repositories with Highest Fork Count')
st.markdown('''
    - :violet[**Diverse Interests:**]  The presence of repositories like :orange["ProgrammingAssignment"] and
    :orange["SpoonKnife"] indicates diverse interests and activities on GitHub. While "ProgrammingAssignment" might be
    related to academic or learning challenges, "SpoonKnife" could be a tool or utility popular among developers.
    - :violet[**Web Development:**]  The :orange["bootstrap"] repository, associated with web development,
    has a substantial fork count, indicating its widespread use and contribution in web projects.
    - :violet[**General Purpose Programming:**] The :orange["Complete Python"] repository suggests a comprehensive
    guide or resource related to Python programming. Its significant fork count reflects the popularity of Python and
    the demand for comprehensive learning resources.
''')
st.bar_chart(forkCount, x="Name", y="Fork Count", color="#ee003f")

# Top 10 Repositories with Most Watchers
watchers = repositoryDataFrame.sort_values(by='Watchers', ascending=False)[:10]
st.subheader('Top 10 Repositories with Most Watchers')
st.markdown('''
    - :violet[**Uniform Popularity:**] Most of the repositories displayed have a fairly consistent number of watchers,
    ranging between 4,000 to 6,000 watchers. This suggests that these repositories are all relatively popular and
    actively monitored by the GitHub community.
    - :violet[**Learning Platforms & Challenges:**] Repositories like :orange["CodeHub"], :orange["Python-100-Days"],
    and :orange["freeCodeCamp"] suggest a strong interest in learning platforms or coding challenges.
    This indicates the continuous demand for educational content and coding challenges on GitHub.
    - :violet[**Machine Learning:**] The :orange["tensorflow"] repository, associated with machine learning, further
    emphasizes the growing interest in AI and machine learning technologies.
''')
st.bar_chart(watchers, x="Name", y="Watchers", color="#0362ff")

# Repositories with the Highest Pull Requests
pullRequests = repositoryDataFrame.sort_values(by='Pull Requests', ascending=False)[:10]
st.subheader('Repositories with Highest Pull Requests')
st.markdown('''
    - :violet[**Active Contribution in Homebrew:**] The repositories :orange["homebrew-cask"] and
    :orange["homebrew-core"] have high pull request counts, suggesting that the Homebrew package manager for macOS is
    actively contributed to and maintained by the community. This reflects its widespread use and significance in the
    macOS developer community.
    - :violet[**Political Data Collection:**] :orange["everypolitician-data"] seems to be a repository related to data
    collection on politicians. Its high pull request count suggests active data updates and contributions, possibly
    indicating a community-driven effort to maintain political data.
''')
st.bar_chart(pullRequests, x="Name", y="Pull Requests", color="#5e16f0")

# Repositories with the Highest Commit Counts
commitCount = repositoryDataFrame.sort_values(by='Commit Count', ascending=False)[:10]
st.subheader('Repositories with Highest Commit Counts')
st.markdown('''
    - :violet[**Linux Dominance:**] The repositories :orange["kernel"], :orange["linux-next"], :orange["linux-mksw"],
    and :orange["mpc-linux-next"] have high commit counts, suggesting that the Linux operating system sees extensive
    development and contributions. The presence of multiple Linux-related repositories underscores the active and
    open-source nature of Linux development.
    - :violet[**Commit Management:**] The repository :orange["Committed"] has a significant commit count.
    It might be related to commit management, version control, or developer tools given its name and high commit count.
    - :violet[**Consistent Activity:**] Most of the repositories displayed have commit counts ranging between 1,000,000
    to 3,000,000, indicating consistent and active development or contributions to these repositories.
''')
st.bar_chart(commitCount, x="Name", y="Commit Count", color="#f67410")

# Repositories with the Highest Issue Counts
issueCount = githubDataFrame.sort_values(by='Issue Count', ascending=False)[:10]
st.subheader('Repositories with Highest Issue Counts')
st.markdown('''
    - :violet[**Aleth's Prominence:**] The orange["aleth"] repository has the highest issue count, considerably
    surpassing the other repositories. This suggests that "aleth", a C++ Ethereum client, is a complex project that
    might have many reported issues, feature requests, and discussions.
    - :violet[**Local Development and Testing:**] orange["localstack"], which provides a local AWS cloud stack for
    testing, has a considerable issue count, indicating its widespread use and the challenges or enhancements requested
    by users.
    - :violet[**Consistency:**] Most repositories, except for orange["aleth"], have issue counts ranging from 100
    to 300, indicating that they have a relatively similar level of activity and engagement.
''')
st.bar_chart(issueCount, x="Repository Name", y="Issue Count", color="#ee003f")
"""
st.divider()
st.markdown('''
    Thank you for the opportunity to work on this fun problem statement!
    
    Built by Snigdha Gupta
    
    Carnegie Mellon University, Tepper School of Business
    
    https://www.linkedin.com/in/snigi/
    
    snigdhag@tepper.cmu.edu
''')