module.exports = ({ env }) => ({
  auth: {
    secret: env('ADMIN_JWT_SECRET', '42297f3d73814b4c3471c49b658d6708'),
  },
});
