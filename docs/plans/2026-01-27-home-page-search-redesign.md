# Home Page Search Redesign

**Date:** 2026-01-27
**Status:** Approved

## Overview

Redesign the Flora Batava Explorer home page to feature prominent search and random plant discovery functionality, establishing it as the primary entry point for plant exploration. Simplify the Plant Detail page by removing duplicate search functionality.

## Goals

1. Make plant search immediately accessible on landing
2. Reduce duplication between home page and Plant Detail page
3. Establish clear user flow: Home → Search/Random → Detail → Explore
4. Maintain informative dataset overview on home page

## Design

### Home Page Structure

The home page will follow this layout hierarchy:

1. **Hero Search Section (New)**
   - Title: "🌿 Flora Batava Plant Explorer"
   - Large, prominent search input with placeholder: "Search by plant ID or name..."
   - Autocomplete dropdown filtering plant_id and Dutch names as user types
   - Display format: `plant_id - Dutch name` (or just plant_id if no Dutch name)
   - "Random Plant" button positioned next to search bar
   - Direct navigation to Plant Detail page on selection

2. **Welcome Text (Existing, condensed)**
   - Brief introduction to the app
   - Overview of main features

3. **Dataset Overview Statistics (Existing)**
   - 4-column metrics: Total Plants, With Taxonomy, Families, Genera
   - Visual Clustering info

4. **Getting Started / About (Existing)**
   - Navigation guide to other pages
   - About the Data section

### Search Interaction Behavior

**Filtering Logic:**
- Real-time filtering as user types (case-insensitive)
- Match against `plant_id` and `Huidige Nederlandse naam` fields
- Show up to 10-15 matches in dropdown
- Display "Showing X of Y matches" if results exceed limit

**Selection & Navigation:**
- User selects plant from autocomplete dropdown
- Set `st.session_state.selected_plant_id`
- Immediately navigate to Plant Detail page using `st.switch_page()`
- No intermediate preview or confirmation

**Random Plant:**
- Single button randomly selects from all plant_ids
- Sets session state and navigates to Plant Detail page
- Same direct navigation behavior as search

**Edge Cases:**
- No matches: Show "No plants found" message
- Empty input: No dropdown shown
- Clear session state on return to home

### Plant Detail Page Simplification

**Removals:**
- Delete entire sidebar search section (lines 20-62)
- Delete "Random plant" button (lines 65-69)
- Delete sidebar separator (line 63)

**Updates:**
- Update fallback message to: "Use the home page to search for a plant, or navigate here from another page"

**Unchanged:**
- All main content (image, taxonomy, colors, clusters, similar/related plants)
- Bottom navigation buttons back to other pages
- Sidebar shows only Streamlit's default page navigation

### Visual Design

**Hero Section:**
- Use `st.columns([3, 1])` for search input and random button layout
- Primary button styling with `type="primary"`
- Visual separation from content below (spacing or subtle styling)

**User Flow:**
- Home page = search hub
- Plant Detail = display/exploration only
- Clear navigation back to home for new searches

## Technical Implementation

### Component Reuse
- Adapt existing search logic from Plant Detail page (lines 24-61)
- Same matching algorithm, display format, session state handling

### Key Changes

**app.py:**
- Import `random` module
- Add hero search section after title
- Filter `plants_df` based on search query
- Use `st.text_input()` with placeholder
- Use `st.selectbox()` for match display
- Handle "Go to plant" and "Random plant" actions
- Navigate with `st.switch_page("pages/4_Plant_Detail.py")`

**4_Plant_Detail.py:**
- Delete lines 16-69 (sidebar search)
- Update fallback message text

### Session State
- Both pages use same key: `selected_plant_id`
- Plant Detail reads from this key (existing behavior)
- No additional state management needed

## Success Criteria

1. Users can search for plants immediately upon landing
2. Search autocomplete works consistently with existing behavior
3. Random plant discovery is accessible from home page
4. Plant Detail page is simplified and focused on display
5. Navigation flow is intuitive: Home → Detail → Back to Home
6. No duplicate functionality between pages

## Implementation Notes

- Maintain keyboard navigation and accessibility
- Session state persists during browser session
- Back button returns user to home page with functional search
- All existing Plant Detail functionality remains intact
