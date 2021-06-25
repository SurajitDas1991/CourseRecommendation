import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

def load_data(data):
    df=pd.read_csv(data)
    return df

def vectorize_text_to_cosine(data):
    count_vectorizer=cv()
    cv_mat=count_vectorizer.fit_transform(data)
    #Get cosine
    cosine_sim_mat=cosine_similarity(cv_mat)
    return cosine_sim_mat


def main():
    st.title("Course Recommendation App")
    menu=['Home','Recommendation','About']
    choice=st.sidebar.selectbox("Menu",menu)
    df=load_data("data/udemy_course_data.csv")
    
    
    if choice=='Home':
        st.subheader('Home')
        st.dataframe(df.head(10))
        
    elif choice=='Recommendation':
        st.subheader('Recommend Courses')
        search_term=st.text_input("Search")
        num_of_recommendations=st.sidebar.number_input("Number",4,30,7)
        if st.button("Recommendation"):
            if search_term is not None:
                pass
    else:
        st.subheader("About")
        st.text('Built with Streamlit and Pandas')


if __name__ == '__main__':
	main()     