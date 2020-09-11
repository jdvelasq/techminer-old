import numpy as np
import re
from techminer.core import explode


import pandas as pd

import datetime


from techminer.core.extract_words import extract_words
from techminer.core.text import remove_accents
from techminer.core.map import map_
from techminer.core.extract_country_name import extract_country_name

import warnings

warnings.filterwarnings("ignore")


class ScopusImporter:
    def __init__(self, input_file="scopus.csv", output_file="techminer.csv"):
        self.input_file = input_file
        self.output_file = output_file
        self.data = None

    def run(self):

        ##
        ## Load data
        ##
        self.data = pd.read_csv(self.input_file)

        ##
        ## Document ID
        ##
        self.data["ID"] = range(len(self.data))

        ##
        ## Steps
        ##
        self.rename_columns()
        self.remove_accents()
        self.remove_no_author_name_available()
        self.format_author_names()
        self.count_number_of_authors_per_document()
        self.calculate_frac_number_of_documents_per_author()
        self.remove_no_author_id_available()
        self.disambiguate_author_names()
        self.remove_text_in_foreing_languages()
        self.extract_country_names()
        self.extract_country_first_author()
        self.reduce_list_of_countries()
        self.transform_author_keywords_to_lower_case()
        self.transform_index_keywords_to_lower_case()
        self.remove_copyright_mark_from_abstracts()
        self.transform_global_citations_NA_to_zero()
        self.format_abb_source_title()
        self.create_historiograph_id()
        self.create_local_references()
        #  self.extract_title_words()
        #  self.extract_abstract_phrases_and_words()
        #  self.highlight_author_keywords_in_titles()
        #  self.highlight_author_keywords_in_abstracts()
        self.compute_bradford_law_zones()

        ##
        ## Replace blanks by pd.NA
        ##
        self.data = self.data.applymap(
            lambda w: pd.NA if isinstance(w, str) and w == "" else w
        )

        self.data = self.data.applymap(
            lambda w: w.replace(chr(8211), "-") if isinstance(w, str) else w
        )

        ##
        ## Transformer output
        ##
        if self.output_file is None:
            return self.data
        self.data.to_csv(self.output_file, index=False)

        self.logging_info("Finished!!!")

    def logging_info(self, msg):
        print(
            "{} - INFO - {}".format(
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), msg
            )
        )

    def rename_columns(self):

        scopus2tags = {
            "Abbreviated Source Title": "Abb_Source_Title",
            "Abstract": "Abstract",
            "Access Type": "Access_Type",
            "Affiliations": "Affiliations",
            "Art. No.": "Art_No",
            "Author Keywords": "Author_Keywords",
            "Author(s) ID": "Authors_ID",
            "Authors with affiliations": "Authors_with_affiliations",
            "Authors": "Authors",
            "Cited by": "Global_Citations",
            "CODEN": "CODEN",
            "Correspondence Address": "Correspondence_Address",
            "Document Type": "Document_Type",
            "DOI": "DOI",
            "Editors": "Editors",
            "EID": "EID",
            "Index Keywords": "Index_Keywords",
            "ISBN": "ISBN",
            "ISSN": "ISSN",
            "Issue": "Issue",
            "Language of Original Document": "Language_of_Original_Document",
            "Link": "Link",
            "Page count": "Page_count",
            "Page end": "Page_end",
            "Page start": "Page_start",
            "Publication Stage": "Publication_Stage",
            "Publisher": "Publisher",
            "PubMed ID": "PubMed_ID",
            "References": "Global_References",
            "Source title": "Source_title",
            "Source": "Source",
            "Title": "Title",
            "Volume": "Volume",
            "Year": "Year",
        }

        self.logging_info("Renaming and selecting columns ...")
        self.data = self.data.rename(columns=scopus2tags)

    def remove_accents(self):
        self.logging_info("Removing accents ...")
        self.data = self.data.applymap(
            lambda w: remove_accents(w) if isinstance(w, str) else w
        )

    def remove_no_author_name_available(self):

        if "Authors" not in self.data.columns:
            return

        self.logging_info('Removing  "[No author name available]" ...')
        self.data["Authors"] = self.data.Authors.map(
            lambda w: pd.NA if w == "[No author name available]" else w
        )

    def format_author_names(self):

        if "Authors" not in self.data.columns:
            return

        self.logging_info("Formatting author names ...")
        self.data["Authors"] = self.data.Authors.map(
            lambda w: w.replace(",", ";").replace(".", "") if pd.isna(w) is False else w
        )

    def count_number_of_authors_per_document(self):

        if "Authors" not in self.data.columns:
            return

        self.logging_info("Counting number of authors per document...")
        self.data["Num_Authors"] = self.data.Authors.map(
            lambda w: len(w.split(";")) if not pd.isna(w) else 0
        )

    def calculate_frac_number_of_documents_per_author(self):

        if "Authors" not in self.data.columns:
            return

        self.logging_info("Counting frac number of documents per author...")
        self.data["Frac_Num_Documents"] = self.data.Authors.map(
            lambda w: 1.0 / len(w.split(";")) if not pd.isna(w) else 0
        )

    def remove_no_author_id_available(self):

        if "Authors_ID" not in self.data.columns:
            return

        self.data["Authors_ID"] = self.data.Authors_ID.map(
            lambda w: pd.NA if w == "[No author id available]" else w
        )

    def disambiguate_author_names(self):

        if "Authors" not in self.data.columns or "Authors_ID" not in self.data.columns:
            return

        self.logging_info("Disambiguate author names ...")

        self.data["Authors"] = self.data.Authors.map(
            lambda w: w[:-1] if not pd.isna(w) and w[-1] == ";" else w
        )

        self.data["Authors_ID"] = self.data.Authors_ID.map(
            lambda w: w[:-1] if not pd.isna(w) and w[-1] == ";" else w
        )

        data = self.data[["Authors", "Authors_ID"]]
        data = data.dropna()

        data["*info*"] = [(a, b) for (a, b) in zip(data.Authors, data.Authors_ID)]

        data["*info*"] = data["*info*"].map(
            lambda w: [
                (u.strip(), v.strip()) for u, v in zip(w[0].split(";"), w[1].split(";"))
            ]
        )

        data = data[["*info*"]].explode("*info*")
        data = data.reset_index(drop=True)

        names_ids = {}
        for idx in range(len(data)):

            author_name = data.at[idx, "*info*"][0]
            author_id = data.at[idx, "*info*"][1]

            if author_name in names_ids.keys():

                if author_id not in names_ids[author_name]:
                    names_ids[author_name] = names_ids[author_name] + [author_id]
            else:
                names_ids[author_name] = [author_id]

        ids_names = {}
        for author_name in names_ids.keys():
            suffix = 0
            for author_id in names_ids[author_name]:
                if suffix > 0:
                    ids_names[author_id] = author_name + "(" + str(suffix) + ")"
                else:
                    ids_names[author_id] = author_name
                suffix += 1

        self.data["Authors"] = self.data.Authors_ID.map(
            lambda z: ";".join([ids_names[w.strip()] for w in z.split(";")])
            if not pd.isna(z)
            else z
        )

    def remove_text_in_foreing_languages(self):

        if "Title" not in self.data.columns:
            return

        self.logging_info("Removing part of titles in foreing languages ...")
        self.data["Title"] = self.data.Title.map(
            lambda w: w[0 : w.find("[")] if pd.isna(w) is False and w[-1] == "]" else w
        )

    def extract_country_names(self):

        if "Affiliations" not in self.data.columns:
            return

        self.logging_info("Extracting country names ...")
        self.data["Countries"] = map_(self.data, "Affiliations", extract_country_name)

    def extract_country_first_author(self):

        if "Countries" not in self.data.columns:
            return

        self.logging_info("Extracting country of first author ...")
        self.data["Country_1st_Author"] = self.data.Countries.map(
            lambda w: w.split(";")[0] if isinstance(w, str) else w
        )

    def reduce_list_of_countries(self):

        if "Countries" not in self.data.columns:
            return

        self.logging_info("Reducing list of countries ...")
        self.data["Countries"] = self.data.Countries.map(
            lambda w: ";".join(set(w.split(";"))) if isinstance(w, str) else w
        )

    def transform_author_keywords_to_lower_case(self):

        if "Author_Keywords" not in self.data.columns:
            return

        self.logging_info("Transforming Author Keywords to lower case ...")
        self.data["Author_Keywords"] = self.data.Author_Keywords.map(
            lambda w: w.lower() if not pd.isna(w) else w
        )
        self.data["Author_Keywords"] = self.data.Author_Keywords.map(
            lambda w: ";".join(sorted([z.strip() for z in w.split(";")]))
            if not pd.isna(w)
            else w
        )

    def transform_index_keywords_to_lower_case(self):

        if "Index_Keywords" not in self.data.columns:
            return

        self.logging_info("Transforming Index Keywords to lower case ...")
        self.data["Index_Keywords"] = self.data.Index_Keywords.map(
            lambda w: w.lower() if not pd.isna(w) else w
        )
        self.data["Index_Keywords"] = self.data.Index_Keywords.map(
            lambda w: ";".join(sorted([z.strip() for z in w.split(";")]))
            if not pd.isna(w)
            else w
        )

    def remove_copyright_mark_from_abstracts(self):

        if "Abstract" not in self.data.columns:
            return

        self.logging_info("Removing copyright mark from abstract ...")
        self.data.Abstract = self.data.Abstract.map(
            lambda w: w[0 : w.find("\u00a9")] if not pd.isna(w) else w
        )

    def transform_global_citations_NA_to_zero(self):

        if "Global_Citations" not in self.data.columns:
            return

        self.logging_info("Removing <NA> from Global_Citations field ...")
        self.data["Global_Citations"] = self.data["Global_Citations"].map(
            lambda w: 0 if pd.isna(w) else w
        )

    def format_abb_source_title(self):

        if "Abb_Source_Title" not in self.data.columns:
            return

        self.logging_info("Removing '.' from Abb_Source_Title field ...")
        self.data["Abb_Source_Title"] = self.data["Abb_Source_Title"].map(
            lambda w: w.replace(".", "") if isinstance(w, str) else w
        )

    def create_historiograph_id(self):

        if "Global_References" not in self.data.columns:
            return

        self.logging_info("Generating historiograph ID ...")
        self.data = self.data.assign(
            Historiograph_ID=self.data.Year.map(str)
            + "-"
            + self.data.groupby(["Year"], as_index=False)["Authors"].cumcount().map(str)
        )

    def create_local_references(self):

        if "Historiograph_ID" not in self.data.columns:
            return

        self.logging_info("Extracting local references ...")

        self.data["Local_References"] = [[] for _ in range(len(self.data))]

        for i_index, _ in enumerate(self.data.Title):

            title = self.data.Title[i_index].lower()
            year = self.data.Year[i_index]

            for j_index, references in enumerate(self.data.Global_References.tolist()):

                if pd.isna(references) is False and title in references.lower():

                    for reference in references.split(";"):

                        if title in reference.lower() and str(year) in reference:

                            self.data.at[j_index, "Local_References"] += [
                                self.data.Historiograph_ID[i_index]
                            ]
                            continue

        self.data["Local_References"] = self.data.Local_References.map(
            lambda w: pd.NA if len(w) == 0 else w
        )
        self.data["Local_References"] = self.data.Local_References.map(
            lambda w: ";".join(w), na_action="ignore"
        )

        local_references = self.data[["Local_References"]]
        local_references = local_references.rename(
            columns={"Local_References": "Local_Citations"}
        )
        local_references = local_references.dropna()

        local_references["Local_Citations"] = local_references.Local_Citations.map(
            lambda w: w.split(";")
        )
        local_references = local_references.explode("Local_Citations")
        local_references = local_references.groupby(
            ["Local_Citations"], as_index=True
        ).size()
        self.data["Local_Citations"] = 0
        self.data.index = self.data.Historiograph_ID
        self.data.loc[local_references.index, "Local_Citations"] = local_references
        self.data.index = list(range(len(self.data)))

    def extract_title_words(self):

        if "Title" not in self.data.columns:
            return

        self.logging_info("Extracting title words ...")
        self.data["Title_words"] = extract_words(data=self.data, text=self.data.Title)

    def extract_abstract_phrases_and_words(self):
        #
        def extract_elementary_context(text):
            text = text.split(". ")
            text = [unit for w in text for unit in w.split(":")]
            text = [unit for w in text for unit in w.split("?")]
            text = [unit for w in text for unit in w.split("!")]
            return "//".join(text)

        if "Abstract" not in self.data.columns:
            return

        self.logging_info("Extracting abstract phrases and words ...")

        self.data["Abstract_phrases"] = self.data.Abstract.map(
            lambda w: extract_elementary_context(w), na_action="ignore"
        )

        self.data["Abstract_phrase_words"] = self.data.Abstract_phrases.map(
            lambda w: "//".join(
                [
                    ";".join(extract_words(data=self.data, text=pd.Series(z)))
                    for z in w.split("//")
                ]
            ),
            na_action="ignore",
        )

        self.data["Abstract_phrase_words"] = self.data.Abstract_phrase_words.map(
            lambda w: w[:-2] if w[-2:] == "//" else w,
            na_action="ignore",
        )

        self.data["Abstract_words"] = self.data.Abstract_phrase_words.copy()
        self.data["Abstract_words"] = self.data.Abstract_words.map(
            lambda w: w.replace("//", ";"), na_action="ignore"
        )
        self.data["Abstract_words"] = self.data.Abstract_words.map(
            lambda w: w.split(";"), na_action="ignore"
        )
        self.data["Abstract_words"] = self.data.Abstract_words.map(
            lambda w: ";".join(sorted(set(w))), na_action="ignore"
        )

    def highlight_author_keywords_in_titles(self):
        #
        def replace_keywords(x):
            #
            for keyword in keywords_list:
                x = re.sub(
                    pattern=re.escape(keyword),
                    repl=keyword.upper().replace(" ", "_"),
                    string=x,
                    flags=re.I,
                )
            return x

        if "Title" not in self.data.columns:
            return

        if "Author_Keywords" not in self.data.columns:
            return

        if len(self.data) >= 200:
            return

        self.logging_info("Marking Author Keywords in Titles ...")

        ##
        ## Builds a list of keywords
        ##
        keywords_list = self.data.Author_Keywords.copy()
        keywords_list = keywords_list.dropna()
        keywords_list = keywords_list.map(lambda w: w.split(";"))
        keywords_list = keywords_list.explode()
        keywords_list = keywords_list.map(lambda w: w.upper())
        keywords_list = keywords_list.tolist()

        ##
        ## Replace in titles
        ##
        self.data["Title_HL"] = self.data.Title
        self.data["Title_HL"] = self.data.Title_HL.map(replace_keywords)

    def highlight_author_keywords_in_abstracts(self):
        #
        def replace_keywords(x):
            #
            for keyword in keywords_list:
                x = re.sub(
                    pattern=re.escape(keyword),
                    repl=keyword.upper().replace(" ", "_"),
                    string=x,
                    flags=re.I,
                )
            return x

        if "Abstract" not in self.data.columns:
            return

        if "Author_Keywords" not in self.data.columns:
            return

        if len(self.data) >= 200:
            return

        self.logging_info("Marking Author Keywords in Abstracts ...")

        ##
        ## Builds a list of keywords
        ##
        keywords_list = self.data.Author_Keywords.copy()
        keywords_list = keywords_list.dropna()
        keywords_list = keywords_list.map(lambda w: w.split(";"))
        keywords_list = keywords_list.explode()
        keywords_list = keywords_list.map(lambda w: w.upper())
        keywords_list = keywords_list.tolist()

        ##
        ## Replace in titles
        ##
        self.data["Abstract_HL"] = self.data.Abstract
        self.data["Abstract_HL"] = self.data.Abstract_HL.map(replace_keywords)

    def compute_bradford_law_zones(self):

        ##
        x = self.data.copy()

        self.logging_info("Computing Bradford Law Zones ...")

        ##
        ## Counts number of documents per Source_title
        ##
        x["Num_Documents"] = 1
        x = explode(
            x[
                [
                    "Source_title",
                    "Num_Documents",
                    "ID",
                ]
            ],
            "Source_title",
        )
        m = x.groupby("Source_title", as_index=False).agg(
            {
                "Num_Documents": np.sum,
            }
        )
        m = m[["Source_title", "Num_Documents"]]
        m = m.sort_values(["Num_Documents"], ascending=False)
        m["Cum_Num_Documents"] = m.Num_Documents.cumsum()
        dict_ = {
            source_title: num_documents
            for source_title, num_documents in zip(m.Source_title, m.Num_Documents)
        }

        ##
        ## Number of source titles by number of documents
        ##
        g = m[["Num_Documents"]]
        g.loc[:, "Num_Source_titles"] = 1
        g = g.groupby(["Num_Documents"], as_index=False).agg(
            {
                "Num_Source_titles": np.sum,
            }
        )
        g["Total_Num_Documents"] = g["Num_Documents"] * g["Num_Source_titles"]
        g = g.sort_values(["Num_Documents"], ascending=False)
        g["Cum_Num_Documents"] = g["Total_Num_Documents"].cumsum()

        ##
        ## Bradford law zones
        ##
        bradford_core_sources = int(len(self.data) / 3)
        g["Bradford_Law_Zone"] = g["Cum_Num_Documents"]
        g["Bradford_Law_Zone"] = g.Bradford_Law_Zone.map(
            lambda w: 3
            if w > 2 * bradford_core_sources
            else (2 if w > bradford_core_sources else 1)
        )

        bradford_dict = {
            num_documents: zone
            for num_documents, zone in zip(g.Num_Documents, g.Bradford_Law_Zone)
        }

        ##
        ## Computes bradford zone for each document
        ##
        self.data["Bradford_Law_Zone"] = self.data.Source_title
        self.data["Bradford_Law_Zone"] = self.data.Bradford_Law_Zone.map(
            lambda w: dict_[w], na_action="ignore"
        )
        self.data["Bradford_Law_Zone"] = self.data.Bradford_Law_Zone.map(
            lambda w: bradford_dict[w], na_action="ignore"
        )


def import_scopus(input_file="scopus.csv", output_file="techminer.csv"):
    #
    ScopusImporter(input_file=input_file, output_file=output_file).run()
