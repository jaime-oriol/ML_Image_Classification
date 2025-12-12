# CLAUDE.md - FootballDecoded Blog Development Guide

## Core Philosophy

**Ve paso a paso, uno a uno. Despacio es el camino más rápido. Escribe siempre el código lo más compacto y conciso posible, y que cumpla exactamente lo pedido al 100%. Sin emojis ni florituras. Usa nombres claros y estándar. Incluye solo comentarios útiles y necesarios.**

Antes de realizar cualquier tarea, revisa cuidadosamente el archivo CLAUDE.md.
Ahí encontrarás las directrices de trabajo y la estructura del proyecto que debes seguir.

### Development Principles

- **KISS (Keep It Simple, Stupid)**: Choose straightforward solutions over complex ones
- **YAGNI (You Aren't Gonna Need It)**: Implement features only when needed
- **Fail Fast**: Check for errors early and raise exceptions immediately
- **Single Responsibility**: Each component, function, and module has one clear purpose
- **Dependency Inversion**: High-level components depend on abstractions, not implementations
- **Consistency**: Follow established patterns in the existing codebase

## Project Structure

### Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript with strict mode
- **Styling**: Tailwind CSS 4 with custom design system
- **Content**: Contentlayer + MDX for articles
- **Authentication**: NextAuth.js with Google OAuth
- **Email**: Resend for newsletter and notifications
- **Deployment**: Vercel with custom domain
- **Analytics**: Umami for privacy-focused tracking

### Key Directories

```
footballdecoded/
├── app/                    # Next.js App Router
│   ├── Main.tsx           # Homepage with featured articles
│   ├── about/             # Author bio and experience
│   ├── api/               # API endpoints
│   │   ├── auth/          # NextAuth configuration
│   │   ├── comments/      # Comment system endpoints
│   │   └── newsletter/    # Newsletter subscription
│   ├── blog/              # Blog system with sections
│   │   ├── player-decoded/
│   │   ├── match-analysis/
│   │   └── team-architecture/
│   ├── contact/           # Contact form
│   ├── newsletter/        # Newsletter pages
│   └── tags/              # Tag-based article filtering
├── components/             # Reusable React components
│   ├── ArticleCard.tsx    # Article preview cards
│   ├── AuthButton.tsx     # Authentication UI
│   ├── CommentForm.tsx    # Comment submission
│   ├── Header.tsx         # Site navigation
│   ├── NewsletterForm.tsx # Newsletter subscription
│   └── social-icons/      # Social media icons
├── content/               # Blog content and configuration
│   ├── articles/          # MDX articles by section
│   │   ├── player-decoded/        # Player analysis section
│   │   ├── match-analysis/        # Match analysis section
│   │   └── team-architecture/     # Team architecture section
│   ├── headerNavLinks.ts  # Navigation configuration
│   └── siteMetadata.js    # Global site settings
├── layouts/               # Page layouts
│   └── PostLayout.tsx     # Article layout with TOC
├── public/static/         # Static assets
│   ├── images/            # Article images and assets
│   └── favicons/          # Site icons
└── scripts/               # Utility scripts
    ├── newsletter-manager.mjs  # Newsletter management
    └── rss.mjs                # RSS feed generation
```

## Development Standards

### Code Style

```typescript
// Component naming: PascalCase
interface ArticleCardProps {
  article: Article
  featured?: boolean
  className?: string
}

export default function ArticleCard({
  article,
  featured = false,
  className = ''
}: ArticleCardProps) {
  return (
    <article className={`article-card ${className}`}>
      {/* Component content */}
    </article>
  )
}

// Custom hooks: use prefix
function useArticleData(slug: string) {
  // Hook implementation
}

// Utility functions: camelCase
function formatPublishDate(date: string): string {
  return new Date(date).toLocaleDateString('es-ES')
}

// Constants: UPPER_SNAKE_CASE
const MAX_ARTICLES_PER_PAGE = 6
const NEWSLETTER_API_ENDPOINT = '/api/newsletter/subscribe'
```

### File Naming Conventions

```bash
# Components: PascalCase
ArticleCard.tsx
NewsletterForm.tsx
CommentsList.tsx

# Pages: kebab-case
app/about/page.tsx
app/player-decoded/page.tsx

# Content: kebab-case
nuevo-articulo-ejemplo.mdx
otro-articulo-ejemplo.mdx

# Assets: kebab-case
imagen-ejemplo-articulo.jpg
otro-banner-ejemplo.png

# Utilities and configs: camelCase or kebab-case
siteMetadata.js
newsletter-manager.mjs
```

### Error Handling

```typescript
// API routes: structured error responses
try {
  const result = await processNewsletterSubscription(email)
  return NextResponse.json({ success: true, data: result })
} catch (error) {
  console.error('Newsletter subscription error:', error)
  return NextResponse.json(
    { error: 'Subscription failed', details: error.message },
    { status: 500 }
  )
}

// Client-side: user-friendly error states
const [error, setError] = useState<string | null>(null)

async function handleSubmit(formData: FormData) {
  try {
    setError(null)
    await submitComment(formData)
  } catch (err) {
    setError('No se pudo enviar el comentario. Inténtalo de nuevo.')
  }
}
```

### Content Structure

```typescript
// MDX frontmatter for articles (REQUIRED FIELDS)
---
title: 'Título del Artículo de Ejemplo'
date: '2024-01-15'
section: 'player-decoded' // player-decoded | match-analysis | team-architecture
image: '/static/images/articles/imagen-ejemplo.jpg'
tags: ['táctica', 'análisis', 'ejemplo']
summary: 'Resumen de ejemplo para un artículo del blog'
author: 'Jaime Oriol'
readingTime: '8 min'
featured: false // Optional: highlight on homepage
---

# Article content in MDX format
```

### Section Color Coding & Themes

```css
/* Player Decoded - Blue theme */
.player-decoded {
  --primary: theme(colors.sky.600);
  --primary-light: theme(colors.sky.100);
  --accent: theme(colors.sky.500);
}

/* Match Analysis - Green theme */
.match-analysis {
  --primary: theme(colors.emerald.600);
  --primary-light: theme(colors.emerald.100);
  --accent: theme(colors.emerald.500);
}

/* Team Architecture - Purple theme */
.team-architecture {
  --primary: theme(colors.indigo.600);
  --primary-light: theme(colors.indigo.100);
  --accent: theme(colors.indigo.500);
}
```

## Development Commands

### Package Management (npm)

```bash
# Install dependencies
npm install

# Add new package
npm install package-name

# Development dependency
npm install --save-dev package-name

# Update dependencies
npm update
```

### Common Development Tasks

```bash
# Development
npm run dev              # Start development server (localhost:3000)
npm run build            # Build for production
npm run start            # Start production server

# Code Quality
npm run lint             # ESLint with auto-fix for app, components, layouts
npm run analyze          # Bundle analyzer for performance optimization

# Content & Newsletter
npm run newsletter       # Newsletter management CLI
# - npm run newsletter list     # View subscriber statistics
# - npm run newsletter export   # Export confirmed emails
# - npm run newsletter test     # Test API connection
```

## Content Creation Workflow

### Adding New Articles

1. **Create MDX file** in appropriate section:

   ```bash
   content/articles/player-decoded/nuevo-analisis.mdx
   ```

2. **Add frontmatter** with all required fields
3. **Add article image** to:

   ```bash
   public/static/images/articles/nuevo-analisis.jpg
   ```

4. **Test locally**:

   ```bash
   npm run dev
   ```

5. **Verify build**:
   ```bash
   npm run build
   ```

### MDX Components Available

```mdx
<!-- Technical concepts with semantic highlighting -->

<TechnicalConcept type="metric">xG</TechnicalConcept>
<TechnicalConcept type="role">Mediocentro defensivo</TechnicalConcept>
<TechnicalConcept type="system">4-3-3</TechnicalConcept>

<!-- Callouts for insights -->

<Callout type="insight">Insight clave sobre el análisis táctico</Callout>

<Callout type="warning">Limitación importante de esta métrica</Callout>

<!-- Statistics cards -->

<StatCard
  title="Presión alta exitosa"
  value="78.4%"
  description="Porcentaje de recuperaciones en campo rival"
  trend="up"
/>

<!-- Technical quotes -->

<TechnicalQuote source="Autor Ejemplo">Cita de ejemplo para mostrar el componente</TechnicalQuote>

<!-- Image with caption -->

<Image
  src="/static/images/articles/pressure-map.jpg"
  alt="Mapa de presión del Manchester City"
  width={800}
  height={500}
  caption="Mapa de calor mostrando las zonas de presión del City"
/>
```

## Branch Strategy & Git Workflow

### Branch Naming Conventions

```
main (protected branch)
  ├── content/add-nuevo-articulo
  ├── feature/improve-newsletter-ui
  ├── fix/mobile-navigation-bug
  ├── style/update-article-cards
  └── docs/update-api-documentation
```

- `content/` - New articles or content updates
- `feature/` - New functionality
- `fix/` - Bug fixes
- `style/` - Design/styling improvements
- `docs/` - Documentation updates

### Workflow Steps

```bash
# 1. Start new task - always from main
git checkout main
git pull origin main
git checkout -b content/add-new-player-analysis

# 2. Work on content with incremental commits
git add content/articles/player-decoded/new-analysis.mdx
git commit -m "content: add player analysis structure"
git add public/static/images/articles/new-analysis.jpg
git commit -m "content: add featured image for tactical analysis"

# 3. Keep branch updated with main
git fetch origin
git rebase origin/main

# 4. Push to remote
git push origin content/add-new-player-analysis

# 5. After merge, cleanup
git checkout main
git pull origin main
git branch -d content/add-new-player-analysis
```

### Commit Message Format

Follow conventional commits specification:

```bash
# Format: <type>(<scope>): <subject>

# Types for blog content
content: New articles or content updates
feat: New features or functionality
fix: Bug fixes
style: Design and styling changes
docs: Documentation changes
refactor: Code improvements without changing functionality
perf: Performance improvements
chore: Maintenance tasks

# Examples
git commit -m "content: add new player analysis article"
git commit -m "feat(newsletter): add confirmation email template"
git commit -m "fix(mobile): resolve navigation menu positioning"
git commit -m "style(cards): improve article card hover effects"
git commit -m "docs: update MDX component usage guide"
```

## Configuration Management

### Environment Variables

```env
# NextAuth Configuration
NEXTAUTH_URL=http://localhost:3000  # Change to production URL
NEXTAUTH_SECRET=your-nextauth-secret-key-here

# Google OAuth (Google Console)
GOOGLE_CLIENT_ID=your-google-oauth-client-id
GOOGLE_CLIENT_SECRET=your-google-oauth-client-secret

# Resend Email Service
RESEND_API_KEY=your-resend-api-key-for-newsletter

# Umami Analytics (optional)
UMAMI_WEBSITE_ID=your-umami-tracking-id
UMAMI_URL=your-umami-instance-url

# Content Configuration
NEXT_PUBLIC_SITE_URL=http://localhost:3000  # Production: https://footballdecoded.com
```

### Site Metadata Configuration

```javascript
// content/siteMetadata.js
const siteMetadata = {
  title: 'FootballDecoded',
  author: 'Jaime Oriol Goicoechea',
  headerTitle: 'FootballDecoded',
  description:
    'Análisis táctico avanzado, métricas cuantitativas y scouting funcional para el fútbol profesional',
  language: 'es-ES',
  theme: 'system',
  siteUrl: 'https://footballdecoded.com',
  siteLogo: '/static/images/logo.png',
  socialBanner: '/static/images/twitter-card.png',
  email: 'joriolgo@gmail.com',
  github: 'https://github.com/jaime-oriol',
  twitter: 'https://x.com/_orio1',
  linkedin: 'https://www.linkedin.com/in/jaime-oriol-goicoechea-801313276/',
  locale: 'es-ES',
  newsletter: {
    provider: 'resend',
  },
  comments: {
    provider: 'giscus', // or 'disqus'
    giscusConfig: {
      // Configuration
    },
  },
}
```

## Newsletter System

### Management Commands

```bash
# View subscriber statistics
npm run newsletter list

# Export confirmed email addresses
npm run newsletter export

# Test Resend API connection
npm run newsletter test

# Custom newsletter management
node scripts/newsletter-manager.mjs --help
```

### Newsletter Workflow

1. **Subscription**: Users subscribe via NewsletterForm component
2. **Confirmation**: Email sent via Resend with confirmation link
3. **Management**: Use CLI tools to manage subscriber list
4. **Export**: Regular export for external email marketing tools

## Security Considerations

### API Security

```typescript
// Rate limiting for API endpoints
import { Ratelimit } from '@upstash/ratelimit'

const ratelimit = new Ratelimit({
  redis: kv,
  limiter: Ratelimit.slidingWindow(5, '1 m'), // 5 requests per minute
})

// Input validation
import { z } from 'zod'

const newsletterSchema = z.object({
  email: z.string().email('Email inválido'),
  name: z.string().min(2, 'Nombre requerido'),
})
```

### Content Security

- Never expose API keys in client-side code
- Validate all user inputs (comments, newsletter)
- Sanitize MDX content before rendering
- Use environment variables for sensitive data
- Regular dependency updates for security patches

### Privacy Compliance

- Newsletter: Double opt-in confirmation required
- Comments: Optional authentication, no required personal data
- Analytics: Privacy-focused (Umami), no personal data collection
- Cookies: Only essential cookies, clear cookie policy

## Performance Optimization

### Image Optimization

```typescript
// Use Next.js Image component
import Image from 'next/image'

<Image
  src="/static/images/articles/player-analysis.jpg"
  alt="Player analysis diagram"
  width={800}
  height={400}
  priority // For above-the-fold images
  placeholder="blur" // Optional: blur placeholder
/>
```

### Bundle Optimization

```bash
# Analyze bundle size
npm run analyze

# Key optimization areas:
# - Code splitting at route level (automatic with App Router)
# - Dynamic imports for heavy components
# - Image optimization with Next.js Image
# - CSS purging with Tailwind
```

## Testing Strategy

### Content Testing

```bash
# Build test - ensures no compilation errors
npm run build

# Link checking (manual)
# - Verify all internal links work
# - Check external links periodically
# - Test newsletter signup flow
# - Verify comment submission
```

### Cross-browser Testing

- Test on Chrome, Firefox, Safari, Edge
- Mobile testing on iOS Safari and Android Chrome
- Verify responsive design at different breakpoints
- Test newsletter and comment forms

## Deployment

### Vercel Configuration

```javascript
// next.config.js key settings
/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  images: {
    domains: ['footballdecoded.com'],
  },
  async redirects() {
    return [
      // Add redirects as needed
    ]
  },
}
```

### Deployment Checklist

1. **Environment Variables**: Set all required vars in Vercel dashboard
2. **Domain Configuration**: Point footballdecoded.com to Vercel
3. **Analytics**: Configure Umami tracking
4. **Email**: Verify Resend configuration
5. **Content**: Test article rendering and newsletter signup
6. **Performance**: Run Lighthouse audit
7. **SEO**: Verify meta tags, sitemap, robots.txt

### Monitoring

- **Vercel Analytics**: Monitor performance and Core Web Vitals
- **Umami**: Track page views and user engagement
- **Resend Dashboard**: Monitor email delivery rates
- **Manual Testing**: Regular content and feature testing

## Claude Code Configuration

### Initial Setup

```bash
# Skip permission prompts for faster workflow
claude --dangerously-skip-permissions

# Configure terminal for better experience
/terminal-setup

# Clear chat between different tasks
/clear
```

### Best Practices

**File Operations**

- Shift+drag to reference files (not regular drag)
- Control+V to paste images (not Command+V)
- Use `@filename` to reference specific files

**Chat Management**

- Queue multiple prompts for batch processing
- Escape to stop Claude (not Control+C)
- Escape twice to see message history
- Up arrow to navigate previous commands

**Project Context (CLAUDE.md)**

- Root `CLAUDE.md` provides project overview and standards
- Always reference these guidelines before starting work
- Update this file when project conventions change

## Contributing Guidelines

### Code Review Criteria

- **Functionality**: Does the code work as intended?
- **Performance**: Are there any performance implications?
- **Security**: Any security vulnerabilities or best practices violations?
- **Accessibility**: Does the UI meet accessibility standards?
- **SEO**: Are meta tags and structured data correct?
- **Content Quality**: Is written content clear and professional?
- **Code Style**: Does it follow the established conventions?

### Review Process

1. **Self-review**: Test locally, check console for errors
2. **Content Review**: Verify article formatting and links
3. **Performance Check**: Run build and check for warnings
4. **Mobile Test**: Verify responsive design
5. **SEO Check**: Verify meta tags and descriptions
6. **Accessibility**: Check color contrast and keyboard navigation

---

**Remember**: This guide is the single source of truth for FootballDecoded development. Keep it updated as the project evolves. When using Claude Code, reference this guide for consistent development practices and maintain the high quality standards expected for a professional football analysis blog.
