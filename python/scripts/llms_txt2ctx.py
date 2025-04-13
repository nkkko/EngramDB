#!/usr/bin/env python3
"""
Convert llms.txt files to XML context for LLMs.

This script parses a llms.txt file and converts it to XML context format,
similar to the llms_txt2ctx CLI from the llms-txt package.
"""

import re
import argparse
import sys
import itertools
from pathlib import Path
import xml.sax.saxutils as saxutils

def chunked(it, chunk_sz):
    it = iter(it)
    return iter(lambda: list(itertools.islice(it, chunk_sz)), [])

def parse_llms_txt(txt):
    """Parse llms.txt file contents in `txt` to a `dict`"""
    def _p(links):
        link_pat = r'-\s*\[(?P<title>[^\]]+)\]\((?P<url>[^\)]+)\)(?::\s*(?P<desc>.*))?'
        links = links.strip()
        if not links:
            return []
        return [re.search(link_pat, l).groupdict()
                for l in re.split(r'\n+', links) if l.strip() and l.strip().startswith('-')]

    sections = {}
    
    # Extract code blocks
    code_blocks = {}
    for i, (code, lang) in enumerate(re.findall(r'```(\w*)\n(.*?)```', txt, re.DOTALL)):
        placeholder = f"__CODE_BLOCK_{i}__"
        code_blocks[placeholder] = {'lang': lang, 'code': code}
        txt = txt.replace(f"```{lang}\n{code}```", placeholder)
    
    # Split into sections
    start, *rest = re.split(r'^##\s*(.*?)$', txt, flags=re.MULTILINE)
    
    for sec_title, sec_content in chunked(rest, 2):
        # If section starts with bullet points, parse as links
        if re.search(r'^\s*-\s*\[', sec_content.strip(), re.MULTILINE):
            sections[sec_title] = _p(sec_content)
        else:
            # Otherwise, store as plain text
            sections[sec_title] = sec_content.strip()
    
    # Extract title and summary
    title_pattern = r'^#\s*(?P<title>.+?)$\n+(?:^>\s*(?P<summary>.+?)$)?'
    title_match = re.search(title_pattern, start.strip(), re.MULTILINE)
    
    if title_match:
        title = title_match.group('title')
        summary = title_match.group('summary') or ""
        
        # Extract info section (content after title/summary and before first ##)
        info_pattern = r'(?:^>\s*.+?$\n+)?(.*)'
        info_match = re.search(info_pattern, start.strip(), re.MULTILINE | re.DOTALL)
        info = info_match.group(1).strip() if info_match else ""
    else:
        title = ""
        summary = ""
        info = start.strip()
    
    # Restore code blocks
    for placeholder, block in code_blocks.items():
        info = info.replace(placeholder, f"```{block['lang']}\n{block['code']}```")
        for sec_title in sections:
            if not isinstance(sections[sec_title], list):
                sections[sec_title] = sections[sec_title].replace(
                    placeholder, f"```{block['lang']}\n{block['code']}```"
                )
    
    result = {
        'title': title,
        'summary': summary,
        'info': info,
        'sections': sections
    }
    
    return result

def create_ctx(txt, include_optional=False):
    """Create XML context from llms.txt content"""
    parsed = parse_llms_txt(txt)
    
    # Escape XML characters in the summary
    escaped_summary = saxutils.escape(parsed['summary'])
    
    xml_parts = [
        f'<project title="{parsed["title"]}" summary=\'{escaped_summary}\'>',
        parsed['info'].strip(),
    ]
    
    # Process non-optional sections
    for section, content in parsed['sections'].items():
        if section.lower() == 'optional' and not include_optional:
            continue
            
        if section.lower() == 'examples':
            xml_parts.append(f'<examples>\n{content}\n</examples>')
        elif section.lower() == 'docs' and isinstance(content, list):
            xml_parts.append('<docs>')
            for item in content:
                title = item.get('title', '')
                url = item.get('url', '')
                desc = item.get('desc', '')
                xml_parts.append(f'<doc title="{title}" url="{url}">{desc}</doc>')
            xml_parts.append('</docs>')
        else:
            if isinstance(content, list):
                links_text = '\n'.join([
                    f"- [{link['title']}]({link['url']}): {link.get('desc', '')}"
                    for link in content
                ])
                xml_parts.append(f'<section title="{section}">\n{links_text}\n</section>')
            else:
                xml_parts.append(f'<section title="{section}">\n{content}\n</section>')
    
    xml_parts.append('</project>')
    
    return '\n\n'.join(xml_parts)

def main():
    parser = argparse.ArgumentParser(description="Convert llms.txt to XML context for LLMs")
    parser.add_argument("file", type=str, help="Path to llms.txt file")
    parser.add_argument("--optional", "-o", action="store_true", 
                        help="Include optional section in output")
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File {args.file} not found", file=sys.stderr)
        sys.exit(1)
    
    txt = file_path.read_text()
    ctx = create_ctx(txt, include_optional=args.optional)
    
    print(ctx)

if __name__ == "__main__":
    main()