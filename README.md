# PreprintScout

PreprintScout is a tool designed to help researchers, including student researchers, discover preprints that can inform current and future research endeavors. It provides an effective means to stay updated on recent preprints, track current trends in your field, and explore trends in other disciplines. PreprintScout uses LLMs and NLP methods to create personalized recommendations.

## Features

- **Recommendations**: Receive tailored recommendations for recent academic preprints based on your unique background and interests.
- **LLMs and NLP**: Leverage your preferred LLMs (e.g., OpenAI GPT, Google Gemini) and NLP methods (e.g., cosine similarity) to obtain the most relevant recommendations
- **Interdiscipinary**: Explore research both *near* and *distance* from your home discipline based on your preferences; including preprints from [arxiv](https://arxiv.org), [OSF](https://osf.io/preprints) (which aggregates 25+ preprint services), and [PhilArchive](https://philarchive.org/).


## Installation

```bash
    pip install preprintscout
or
    git clone https://github.com/yourusername/PreprintScout.git

```

## Dependencies
##### arxiv>=2.1.3
##### beautifulsoup4>=4.12.3
##### google-generativeai>=0.7.0
##### langdetect>=1.0.9
##### openai>=1.35.7
##### pandas>=2.2.2
##### pytz>=2024.1
##### Requests>=2.32.3
##### retry>=0.9.2
##### scikit_learn>=1.4.2

## Run the application:
```bash
import preprintscout

preprintscout(your_short_biography, huggingface_api_key, openai_api_key="xxxxxx", google_api_key=None, interdisciplinary="3", output_path="/path/to/output")
```
## Configuration

### Required -- Write a short (150 to 200 words) biographical statement about your research, interests, academic background, and so forth.
### This is used for creating recommendations that will be of interest to you!

```bash
your_short_biography = "I am a professor of engineering management. My research is in the application of artificial intelligence in managing engineering systems for electical vehicles... "
```

#### Required -- For recommendations include an API key from your HuggingFace account. For security you can also store this as an environment variable. For example, "${HF_API_KEY}"
```bash
huggingface_api_key = "xxxxxxxxxxxxxxxxxxx"
```

#### Required -- Must have one, either an OpenAI or Google Gemini API key.
- ##### Optional -- For LLM based recommendations include an API key from your OpenAI account. For security you can also store this as an environment variable. For example, "${OPENAI_API_KEY}"
```bash
openai_api_key =  "xxxxxxxxxxxxxxxxxxx"
```
- ##### Optional -- For LLM based recommendations include an API key from your OpenAI account. For security you can also store this as an environment variable. For example, "${GOOGLE_API_KEY}"
```bash
google_api_key =  "xxxxxxxxxxxxxxxxxxx"
```

#### Optional --  Using 1 to 4, indicate your interest in research outside of your home discipline
- ##### 1 = connected disciplines
- ##### 2 = adjacent disciplines (default)
- ##### 3 = tangential disciplines
- ##### 4 = peripheral disciplines

```bash
interdisciplinary = "3"
```

#### Optional -- You can save JSON copies of recommendations include the path.
```bash
output_path = "/path/to/output"
```
##### This will create the directory if it doesn't exist yet. Be sure that has a the leading slash. For instance, on a Mac you could have ""/Users/your_name/recommendations/preprints"

#### Here is a complete example (note that opetional arguments have to be labeled in the function):
```bash
import preprintscout as pps

your_short_biography = "I am a professor of engineering management. My research is in the application of artificial intelligence in managing engineering systems for electical vehicles... "
huggingface_api_key = "xxxxxxxxxxxxxxxxxxx"

pps(your_short_biography, huggingface_api_key, openai_api_key="xxxxxx", google_api_key=None, interdisciplinary="3", output_path="/path/to/output")
```
## Output
#### The return is a JSON file grouped by type of recommendation, with 5 recommendations for each category. Here an example with one recommendation for each category and reduced descriptions.

- ##### arxiv_llm_recs = recommendations of recent preprints from arxiv.org determined by your selected LLM and based on your profile description.
- ##### osf_phil_llm_recs = recommendations of recent preprints from OSF.io and PhilArchive.org determined by your selected LLM and based on your profile description.
- ##### arxiv_cosine_ranked = recommendations of recent preprints from arxiv.org determined by using cosine-similary of the article abstract and your profile description.
- ##### osf_phil_cosine_ranked = recommendations of recent preprints from OSF.io and PhilArchive.org determined by using cosine-similary of the article abstract and your profile description.
- ##### interdisciplinary_ranked = a reranking of all recent prepreints (arxiv, OSF, and PhilArchive) based on the semanitic distance from your home discipline and your interdiscplinary interest setting.
- ##### score = cosine-similarity of the article abstract and your profile description.
- ##### dissim_value = Semanitic distance between the article abstract and your home discipline.


```bash
{
    "arxiv_llm_recs": [
        {
            "article": [
                "1.2 Computer and information sciences",
                "http://arxiv.org/abs/2406.19334v1",
                "Multi-RIS-Empowered Multiple Access: A Distributed Sum-Rate Maximization Approach",
                "The plethora of wirelessly connected devices, whose deployment density ...",
                "This article presents a new communication scheme for 6G wireless networks..."
            ],
            "dissim_value": 0.120610034
        },

    ],
    "osf_phil_llm_recs": [
        {
            "article": [
                "5.1 Psychology",
                "https://osf.io/preprints/psyarxiv/fqzd2/",
                "Who is We: Capturing (European) Identity Content by Integrating Qualitative Methods in Survey-Based Approaches",
                "European identity can mean different things to different people. Yet, past quantitative research ...",
                "This article presents two methods to assess European identity content that can be implemented in survey research."
            ],
            "dissim_value": 0.455544972
        },

    ],
    "arxiv_cosine_ranked": [
        {
            "article": [
                "1.2 Computer and information sciences",
                "http://arxiv.org/abs/2406.19296v1",
                "Vehicle-to-Grid Technology meets Packetized Energy Management",
                "The global energy landscape is experiencing a significant transformation...",
                "This article presents a co-simulation platform to investigate integration of V2G with PET in microgrid..."
            ],
            "score": 0.18289190091265142,
            "dissim_value": 0.120610034
        },

    ],
    "osf_phil_cosine_ranked": [
        {
            "article": [
                "6.3 Philosophy ethics and religion",
                "https://philarchive.org/rec/JIAAAD",
                "AGGA: A Dataset of Academic Guidelines for Generative AIs.",
                "AGGA (Academic Guidelines for Generative AIs) is a dataset of 80 academic guidelines for the usage of generative AIs and large language models in academia...",
                "The article introduces a dataset of 80 academic guidelines for the usage of generative AIs...",
                "The article introduces a dataset of 80 academic guidelines for the usage of generative AIs..."
            ],
            "score": 0.33518822100626167,
            "dissim_value": 0.559099337
        },

    ],
    "interdisciplinary_ranked": [
        {
            "article": [
                "2.2 Electrical engineering; electronic engineering; information engineering",
                "http://arxiv.org/abs/2406.19305v1",
                "A Max Pressure Algorithm for Traffic Signals Considering Pedestrian Queues",
                "This paper proposes a novel max-pressure (MP) algorithm that incorporates\npedestrian traffic into the MP control architecture. Pedestrians are modeled as\nbeing included in one of two groups: those walking on sidewalks...",
                "This article proposes a novel max-pressure algorithm that incorporates pedestrian traffic..."
            ],
            "score": 0.12583606498802957,
            "dissim_value": 0.1
        },
    ]
}
```

## Usage
- Launch the app and explore recent preprints that are recommended based on your profile.
- Get interdisciplinary recommendations from fields *close* or *far* from your home discipline.
- Save preprints to your personalized output directory for easy access.
- Easily add a daily email of recommended preprint using GitHub Actions (link coming soon)

## Known Issues
Check [Github](https://github.com/ryanrwatkins/preprints_recommender).

## Contributing
Contributions are welcome! Please read our contributing guidelines for more details (coming soon).

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or feedback, please contact me on [Github](https://github.com/ryanrwatkins/preprints_recommender).

## Troubleshooting

#### If you encounter issues importing 'preprintscout' after installing the package, follow these steps to find the installation path and update the 'PYTHONPATH'.

#### Step 1: Verify and find the installation path:
##### Use the '>>> pip show preprintscout' command and note the location. For example, if the location is /Users/yourusername/.local/lib/python3.9/site-packages, you will use this path.

```bash
pip show preprintscout
```

#### Step 2: Update PYTHONPATH:
##### Temporarily update PYTHONPATH in your terminal session:

```bash
export PYTHONPATH=$PYTHONPATH:/Users/yourusername/.local/lib/python3.9/site-packages
```
