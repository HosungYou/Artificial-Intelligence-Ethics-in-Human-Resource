# Converting Manuscript to Word APA 7th Format

## Quick Conversion with Pandoc

Install Pandoc if not already installed:
```bash
# macOS
brew install pandoc

# Windows
choco install pandoc

# Linux
sudo apt-get install pandoc
```

### Basic Conversion
```bash
pandoc main_manuscript_APA7.md -o manuscript_APA7.docx --reference-doc=apa7-template.docx
```

### Using the Provided Script
```bash
chmod +x convert_to_word.sh
./convert_to_word.sh
```

## APA 7th Edition Formatting Checklist

### Title Page
- [ ] Running head (uppercase, flush left)
- [ ] Page number (flush right)
- [ ] Title (bold, centered, title case)
- [ ] Author name(s)
- [ ] Institutional affiliation(s)
- [ ] Author note (if applicable)

### General Formatting
- [ ] Font: 12-point Times New Roman
- [ ] Double-spacing throughout
- [ ] 1-inch margins on all sides
- [ ] Left-aligned text (ragged right edge)
- [ ] First line indent: 0.5 inches

### Abstract
- [ ] Label "Abstract" (bold, centered)
- [ ] Single paragraph, no indentation
- [ ] 150-250 words
- [ ] Keywords on new line, indented, italicized "Keywords:" prefix

### Headings (5 levels)
- Level 1: Centered, Bold, Title Case
- Level 2: Flush Left, Bold, Title Case
- Level 3: Flush Left, Bold Italic, Title Case
- Level 4: Indented, Bold, Title Case, Ending with Period.
- Level 5: Indented, Bold Italic, Title Case, Ending with Period.

### References
- [ ] "References" heading (bold, centered)
- [ ] Hanging indent (0.5 inches)
- [ ] Double-spaced
- [ ] Alphabetized by first author's last name
- [ ] DOIs as hyperlinks

### Tables and Figures
- [ ] Numbered consecutively (Table 1, Figure 1, etc.)
- [ ] Title in italics
- [ ] Notes below table
- [ ] High-resolution images (300+ DPI)

## Post-Conversion Word Formatting

After converting to Word, manually check and adjust:

1. **Fonts**: Ensure Times New Roman 12pt throughout
2. **Spacing**: Set to double-space (2.0)
3. **Margins**: Set to 1 inch all sides
4. **Headers**: Add running head and page numbers
5. **Title page**: Adjust spacing and positioning
6. **Tables**: Format with APA table style
7. **References**: Apply hanging indent

## APA 7th Reference Templates

### Journal Article
Author, A. A., & Author, B. B. (Year). Title of article. *Journal Name*, *Volume*(Issue), Page-Page. https://doi.org/xxxxx

### Book
Author, A. A. (Year). *Title of work: Capital letter also for subtitle*. Publisher. https://doi.org/xxxxx

### Chapter in Edited Book
Author, A. A. (Year). Title of chapter. In E. E. Editor (Ed.), *Title of book* (pp. xx-xx). Publisher.

### Conference Paper
Author, A. A. (Year, Month Day-Day). *Title of paper*. Conference Name, Location.

### Online Document
Author, A. A. (Year, Month Day). *Title of document*. Site Name. URL

---

*For detailed APA 7th guidelines, refer to:*
*Publication Manual of the American Psychological Association (7th ed.)*
