# PreprintScout

PreprintScout is a tool designed to help researchers and educators discover and manage preprints in their fields. It provides a useful tool for exploring recent preprints, tracking updates, and curating a personalized library.

## Features

- **Search and Discovery**: Searches for preprints using various filters.
- **Notifications**: Get updates on new preprints in specified research areas.
- **Collaboration**: Share curated lists of preprints with colleagues.

## Installation

Clone the repository:

```bash
   git clone https://github.com/yourusername/PreprintScout.git
   cd PreprintScout
```
## Install the required dependencies:
```bash
pip install -r requirements.txt
```
## Run the application:
```bash
from preprintscout import recommendation_main as pps

pps(your_short_biography, adjacent_interests, huggingface_api_key, openai_api_key, output_path)
```
## Configuration
You can configure various settings in the config.yaml file:

If you want to save a copy of the recommendations, create a directory and include the path to it.

### Write a short (150 to 200 words) biographical statement about your research, interests, academic background, and so forth.
### This is used for creating recommendations that will be of interest to you!
your_short_biography: "I am a professor of engineering management. My research is in the application of artificial intelligence in managing engineering systems for electical vehicles... "

### Using 1 to 4, Indicate your interest in research further from your home discipline
#### 1 = connected disciplines
#### 2 = adjacent disciplines
#### 3 = tangential disciplines
#### 4 = peripheral disciplines
adjacent_interests: "3"

### For recommendations include an API key from your HuggingFace account.
### For security you can also store this as an environment variable. For example, "${HF_API_KEY}"
huggingface_api_key: "xxxxxxxxxxxxxxxxxxx"

### For LLM based recommendations include an API key from your OpenAI account.
### For security you can also store this as an environment variable. For example, "${OPENAI_API_KEY}"
openai_api_key:  "xxxxxxxxxxxxxxxxxxx"

### Optional, if you want to story copies of recommendations create directory and put the path here.
output_path: "/path/to/output"

It will create the directory if it doesn't exist yet. Be sure that has a the leading slash. For instance, on a Mac you could have ""/Users/your_name/recommendations/preprints"


## Usage
Launch the app and explore preprints using the search feature.
Use filters to narrow down results based on your research interests.
Save preprints to your personalized library for easy access.
Enable notifications to stay updated on new releases.
Customize forms to better suit your data collection needs.


## Known Issues
Currently, there's no straightforward way to gather auto routes.
## Contributing
Contributions are welcome! Please read our contributing guidelines for more details.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or feedback, please contact your email.
