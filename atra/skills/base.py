from typing import List, Dict
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from atra.text_utils.embedding import generate_embedding
import langdetect
from atra.text_utils.translation import translate

class BaseSkill:
    def __init__(
        self,
        name: str,
        description: str,
        entities: Dict[str, str],
        examples: List[str],
        module: callable,
    ):
        """
        Base class for all skills.
        :param name: Name of the skill
        :param description: Description of the skill
        :param entities: Entities that the skill can handle, the key is the name of the entity
        and needs to be the named argument of the module
        :param examples: Examples of how to use the skill

        Example:
        >>> from atra.skills.base import BaseSkill
        >>> skill = BaseSkill(
            name="Wikipedia Skill",
            description="This skill uses Wikipedia to generate short summaries about a given topic.",
            entities={"query": "extract the search-query from the given prompt"},
            examples=["Erzähl mir etwas über Angela Merkel", "Tell me something about Donald Trump"])
        """
        self.name = name
        self.description = description
        self.entities = entities
        self.examples = examples
        self.module = module

    def ask_llm(self, prompt: str, **kwargs) -> str:
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"

        def query(payload):
            response = requests.post(API_URL, json=payload)
            return response.json()

        output = query(
            {
                "inputs": prompt,
            }
        )

        return output[0]["generated_text"]

    def extract_entities(self, prompt: str) -> Dict[str, str]:
        """
        Extracts entities from the given prompt.
        :param prompt: Prompt to extract entities from
        :return: Dictionary of extracted entities

        Example:
        >>> from atra.skills.base import BaseSkill
        >>> skill = BaseSkill(
            name="Wikipedia Skill",
            description="This skill uses Wikipedia to generate short summaries about a given topic.",
            entities={"query": "extract the search-query from the given prompt"},
            examples=["Erzähl mir etwas über Angela Merkel", "Tell me something about Donald Trump"])
        >>> skill.extract_entities("Erzähl mir ertwas über Angela Merkel")
        {"query": "Angela Merkel"}
        """
        extracted_entities = {}
        for entity_name, entity_prompt in self.entities.items():
            template = f"""Input: {prompt}
            Instruction: {entity_prompt}"""
            extracted_entities[entity_name] = self.ask_llm(template)

        return extracted_entities

    def answer(self, prompt) -> str:
        lang = langdetect.detect(prompt)
        entities = self.extract_entities(prompt)
        answer = self.module(**entities)
        lang_answer = langdetect.detect(answer)
        if lang_answer != lang:
            answer = translate(answer, lang_answer, lang)
        
        return answer


class SkillStorage:
    def __init__(self):
        self.skills = []
        self.search_index = QdrantClient(":memory:")
        self.id_to_use = 1

        self.search_index.recreate_collection(
            collection_name="atra_skills",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    def add_skill(self, skill: BaseSkill):
        self.skills.append(skill)
        for example in skill.examples:
            embeddings = generate_embedding(example)
            for embedding in embeddings:
                self.search_index.upsert(
                    collection_name="atra_skills",
                    points=[
                        PointStruct(
                            id=self.id_to_use,
                            vector=embedding.tolist(),
                            payload={"name": skill.name},
                        )
                    ],
                )
                self.id_to_use += 1

    def remove_skill(self, skill: BaseSkill):
        for i, s in enumerate(self.skills):
            if s.name == skill.name:
                self.skills.pop(i)
                return

    def choose_skill(self, prompt: str) -> BaseSkill:
        embeddings = generate_embedding(prompt)
        search_result = self.search_index.search(
            collection_name="atra_skills",
            query_vector=embeddings[0].tolist(),
            filter=None,
            top=1,
        )
        return [
            skill
            for skill in self.skills
            if skill.name == search_result[0].payload["name"]
        ][0], search_result[0].score

    def answer(self, prompt: str) -> str:
        skill, score = self.choose_skill(prompt)
        if score < 0.5:
            return False
        
        return skill.answer(prompt) + f"\n\nThis skill ({skill.name}) was chosen with a score of {score:.2f}."
