module.exports = {
  routes: [
    {
      method: 'GET',
      path: '/ebooks/sync',
      handler: 'ebook.booksync',
      config: {
        auth: false,
      },
    },
  ],
};
