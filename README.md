# Pitch Deck Data Extraction System
## Extraction
### System overview
#### Problem
Bridge Latam was looking for a system to extract data from pitch decks in bulk. There is no easy way to download all
pitch decks and analyze them.
#### Solution
This data extraction system uses Airtable's downloaded CSV. This CSV contains a column of attachments, where all the
pitch decks are stored as links, pointing to Airtable. This allows the system to immediately pass the pitch deck PDFs to
Gemini and the system itself, which in turn extracts all necessary data.
### Gemini
- #### Notes
  - Issues
    - Prompt engineering must be extremely specific. Sometimes Gemini, like all LLMs, hallucinates. Therefore, if the
    prompt is not extremely specific, users may encounter issues.
  - Notes
    - Gemini does not extract all the information. For example, the length of the PDFs is calculated by the code without
    the use of LLMs.
## Data
### Collected Data
The following list provides the data which is attempted to be extracted from the pitch decks:
#### LLM Extraction
- Female founder (Does the startup have at least one female founder?)
- Round size (Size of the funding round)
- Sector (Industry the startup belongs to)
- Headquarters (Location of the startup’s headquarters)
- Valuation (Startup’s valuation)
- Number of letters in the startup’s name
- Number of founders (How many founders the startup has)
- Whether the pitch deck starts with the founding team
- TAM (Total Addressable Market)
- Solo founder? (Does the startup have only one founder?)
- Differentiation by funding stage (Pre-Seed, Seed, Series A, etc.)
- Amount raised in the past
- Date created (When the startup was founded)
#### Algorithm Extraction
- Number of pages in the pitch deck
### Future data
- Age of founders
- Algorithm by industry (Create industry clusters)
- Founder personality profile
### Output
The system outputs data in an **XLS** format. The system initially creates an Excel file where all the data will
be exported. In order for the system to save **RAM**, it actively edits the file, appending new information
until it has finished. The strategy further allows for any errors not to *delete* all the *scraped* data.