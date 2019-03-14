import re

import sys
         
sys.path.append(".")

from util.UserStory import UserStory

user_story_full_pattern = re.compile(r'^(como|yo como)(.*)(\bdeseo\b|\bquiero\b|\bnecesito\b|\brequiero\b|\bme gustaría\b|\bquisiera\b|\bnecesita\b)(.*)(\bpara|con el fin\b)(.*)')
user_story_need_pattern = re.compile(r'^(como|yo como)(.*)(\bdeseo\b|\bquiero\b|\bnecesito\b|\brequiero\b|\bme gustaría\b|\bquisiera\b|\bnecesita\b)(.*)')


def parse_user_story(user_story_text):
    m = user_story_full_pattern.match(user_story_text)
    m = m if m is not None else user_story_need_pattern.match(user_story_text)
    if m is None:
        return None
    groups = m.groups()
    user_story = UserStory(groups)
    return user_story


if __name__ == '__main__':
    parse_user_story("como gerente quiero listar los productos para tener informacion")
