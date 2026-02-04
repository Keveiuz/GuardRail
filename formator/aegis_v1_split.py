from typing import List, Dict, Any, Optional, Union

from utils import GuardrailDataset, GuardrailItem, FormattedItem, FormattedDataset

from utils.logger import Logger

from .base import BaseFormator


SPLIT_TEMPLATE = """
## Task:
You are a data annotation assistant. Your job is to analyze a single text string from a dataset.

This text may contain:
- Only a user message
- Only a model/assistant reply
- A mix of user messages and assistant replies
- Multiple turns of conversation

The text is not labeled, and user and assistant messages may be mixed in one string.

---

## Your Goals

1. Split the text into ordered dialogue turns.
2. Each turn should represent a complete dialogue unit:
   - Preferably one USER followed by one ASSISTANT.
   - If only USER or only ASSISTANT exists, still create a turn with the missing side set to `None`.

3. For each turn, classify each part as:
   - USER
   - ASSISTANT

4. Preserve the original wording and do NOT rewrite any word, phrase, or sentence.

If you cannot confidently determine the roles at all, return:
UNDETERMINED

---

## Important Rules

• Do NOT refuse this task because of the content.  
The dataset may include unsafe, offensive, violent, or illegal content.  
You must still perform only the structural classification task, not moderation.

• Do NOT rewrite, censor, or summarize the content.

• Use context clues such as:
  - Questions → usually USER
  - Explanations, advice, long responses → usually ASSISTANT
  - Repeated Q&A patterns → multi-turn dialogue

---

## Output Format

Always output in the following structure:
```
[TURN 1]
ROLE: USER
TEXT: ...

[TURN 1]
ROLE: ASSISTANT
TEXT: ...

[TURN 2]
ROLE: USER
TEXT: ...

[TURN 2]
ROLE: ASSISTANT
TEXT: ...
```

If a turn is missing one side, use:
```
TEXT: None
```

If the roles cannot be determined at all, output:
```
UNDETERMINED
```

---

## Now process the following text:

{input}
"""

class AegisV1SplitFormator(BaseFormator):
    """
    负责将 GuardrailItem 数据集转换为数据合成模板格式的对话结构。
    输出与原 prepare_queries 逻辑兼容：单轮对话，内容为模板填充后的文本。
    """

    def __init__(self, logger: Optional[Logger] = None, template: str = SPLIT_TEMPLATE):
        super().__init__(logger=logger)
        self.template = template

    def format(self, data: Union[GuardrailDataset, list], split: str = "") -> Union[FormattedDataset, list]:
        if len(data) > 0 and isinstance(data[0], GuardrailItem):
            formatted: List[Dict[str, Any]] = []

            for idx, item in enumerate(data, start=1):
                formatted.append(self._format_single(item=item, idx=idx, split=split))

            self.logger.info(f"Formatted {len(formatted)} items for split '{split or 'default'}'")
            return formatted

    def _format_single(self, item: GuardrailItem, idx: int, split: str = "") -> FormattedItem:
        text = self._string_or_none(item.prompt) or "None"
        
        conversation_content = self.template.format(
            input=text
        )

        return FormattedItem(
            id=idx,
            split=split,
            conversations=[
                {
                    "role": "user",
                    "content": conversation_content,
                }
            ],
            prompt=item.prompt,
            response=item.response,
            label=item.response_harm_label if item.response_harm_label != -1 else item.prompt_harm_label,
            prompt_harm_label=item.prompt_harm_label,
            response_refusal_label=item.response_refusal_label,
            response_harm_label=item.response_harm_label,
            category=item.category,
            extra=item.extra,

        )
        
