# web visualization

Now using vsc Live Server for basic prototyping, and reached out to
NSSTC IT about a CGI-capable intranet or publicl web space.

## bootstrap reminders
 - Containers are the highest-level layout element in bootstrap, and
   may be responsive (with max-width changing at each break point)
   or fluid-width (always 100% width).
    - These allow for centering and horizontally padding contents.
 - Breakpoints are based on `min-width` media queries; bootstrap
   supports (None [extra small], sm, md, lg, xl, xxl).
 - Rows and columns are classes of divs. Rows wrap columns; columnns
   have padding (interior) used to control their apparent separation.
   Rows have negative margins to account for alignment.
 - Content must be placed within columns that are first-order
   children of rows.
 - [Media queries][1] are a CSS feature applying styles based on
   browser or operating system properties using "at-rules"
 - .nav is a class for the `<ul/>` tag for navigation components with
   items having nice flexbox properties and enable/disable, etc.
 - .dropdown-menu class of `<div/>` can be embedded within a
   .nav-item `<li/>` element having the .dropdown class.

[1]:https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_media_queries/Using_media_queries
