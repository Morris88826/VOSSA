import { createRouter, createWebHistory } from 'vue-router'
import ComparisonView from '../ComparisonView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: ComparisonView
    }
  ],
})

export default router
